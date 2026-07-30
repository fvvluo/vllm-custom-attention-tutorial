#!/usr/bin/env python
"""
bench_decode_tpot.py —— decode TPOT 微基准（kernel-level，脱离 vLLM 服务）
============================================================================

目的：**可量化**地对比 decode 各优化对 TPOT 的影响，无需起 32B 服务（快、抗 GPU 抢占）。
直接对 custom_backend.triton_attention.paged_attention_triton 的 **decode 路径**（q_len==1）
计时，覆盖 Qwen3-32B 维度（num_heads=64, num_kv_heads=8, head_size=128, block_size=16）。

对比的配置（每个作为独立子进程，用 env 切换，保证 kernel 以对应 launch 参数重编译）：
  1. baseline      : split-KV 关（CUSTOM_DEC_SPLITS=1）——模拟"优化前"逐 CTA 串行扫 KV
  2. +splitKV      : split-KV 自动（现状默认，warps/stages 默认）
  3. +splitKV+tuned: split-KV + 最优 num_warps/num_stages（本次调参）
  4. fp8KV         : 预量化 e4m3 KV 常驻（dense，半字节/步，近无损）
  5. sparse0.25    : bf16 KV + 只读 25% top-k 块（近似）

指标（每层单 attention）：
  - ms/step        : 单次 decode kernel 平均耗时（1 层）
  - TPOT_attn(ms)  : ms/step × num_layers（默认 64），即端到端里"纯 attention"部分的每 token 耗时
  - KV 带宽(TB/s)  : 每步读的 KV 字节 / 时间（dense 读全 KV；fp8 半字节；sparse 读 sparsity 比例）
  - speedup        : 相对 baseline 的加速

用法：
  # 驱动模式（默认）：自动 spawn 各配置子进程，汇总打印 markdown 表
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python scripts/bench_decode_tpot.py --gpu 6

  # 单配置 worker 模式（内部用，也可手动跑单点）
  ... python scripts/bench_decode_tpot.py --worker --ctx 98304 --bs 1 --path fp8KV
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NUM_LAYERS = 64          # Qwen3-32B decoder 层数（把单层 kernel 耗时换算成端到端 attention TPOT）
NH, NKV, HS, BS = 64, 8, 128, 16   # Qwen3-32B attention 维度

# 每个 path 对应的 env（子进程导入 triton_attention 时生效）
# 说明：warps/stages 已单独扫参，结论是 decode 带宽受限、调参中性（见 README/报告），
# 故这里用最终默认 warps=4 stages=2，不再单列 "+tuned"。
PATH_ENV = {
    "baseline":       {"CUSTOM_DEC_SPLITS": "1", "CUSTOM_DEC_WARPS": "4", "CUSTOM_DEC_STAGES": "2"},
    "+splitKV":       {"CUSTOM_DEC_SPLITS": "0", "CUSTOM_DEC_WARPS": "4", "CUSTOM_DEC_STAGES": "2"},
    "fp8KV":          {"CUSTOM_DEC_SPLITS": "0", "CUSTOM_DEC_WARPS": "4", "CUSTOM_DEC_STAGES": "2"},
    "sparse0.25":     {"CUSTOM_DEC_SPLITS": "0", "CUSTOM_DEC_WARPS": "4", "CUSTOM_DEC_STAGES": "2",
                       "CUSTOM_SPARSE": "1", "CUSTOM_SPARSITY": "0.25", "CUSTOM_SPARSE_MIN_LEN": "2048"},
}
PATHS = list(PATH_ENV.keys())


# ---------------------------------------------------------------------------
# worker：在指定 env 下测一个 (ctx, bs, path) 点，打印一行 JSON
# ---------------------------------------------------------------------------
def build_decode_inputs(ctx, bs, dev, dtype):
    """构造纯 decode 输入：bs 条请求，各 context=ctx，本步 1 个 query token。"""
    import torch
    torch.manual_seed(0)
    seqs = [(ctx, 1)] * bs
    kv = []
    for (sl, _ql) in seqs:
        k = torch.randn(sl, NKV, HS, device=dev, dtype=dtype)
        v = torch.randn(sl, NKV, HS, device=dev, dtype=dtype)
        kv.append((k, v))
    blocks_per_seq = [(sl + BS - 1) // BS for (sl, _) in seqs]
    total_blocks = sum(blocks_per_seq) + 2
    max_num_blocks = max(blocks_per_seq)
    key_cache = torch.zeros((total_blocks, NKV, BS, HS), device=dev, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((bs, max_num_blocks), device=dev, dtype=torch.int32)
    nb = 0
    for si, (k, v) in enumerate(kv):
        for j in range(len(k)):
            lb, slot = j // BS, j % BS
            if slot == 0:
                block_table[si, lb] = nb; nb += 1
            pb = block_table[si, lb]
            key_cache[pb, :, slot, :] = k[j]
            value_cache[pb, :, slot, :] = v[j]
    q = torch.randn(bs, NH, HS, device=dev, dtype=dtype)   # 每条请求 1 token
    qsl = torch.arange(bs + 1, device=dev, dtype=torch.int32)  # query_len 全 1
    seq_lens = torch.tensor([sl for sl, _ in seqs], device=dev, dtype=torch.int32)
    tsi = torch.arange(bs, device=dev, dtype=torch.int32)      # token i -> seq i
    scale = 1.0 / HS ** 0.5
    return dict(query=q, key_cache=key_cache, value_cache=value_cache,
                query_start_loc=qsl, seq_lens=seq_lens, token_seq_idx=tsi,
                block_table=block_table, scale=scale, max_num_blocks=max_num_blocks)


def _bench(fn, warmup=10, iters=50):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters   # ms


def worker(ctx, bs, path, gpu):
    import torch
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"; dtype = torch.bfloat16
    from custom_backend.triton_attention import paged_attention_triton  # env 已在导入前设好

    m = build_decode_inputs(ctx, bs, dev, dtype)
    out = torch.empty(bs, NH, HS, device=dev, dtype=dtype)
    is_fp8 = (path == "fp8KV")

    if is_fp8:
        e4m3 = torch.float8_e4m3fn
        kc, vc = m["key_cache"], m["value_cache"]
        k_scale = (kc.abs().amax().clamp_min(1e-12) / 448.0).float().view(1)
        v_scale = (vc.abs().amax().clamp_min(1e-12) / 448.0).float().view(1)
        kc_fp8 = (kc.float() / k_scale).clamp(-448, 448).to(e4m3)
        vc_fp8 = (vc.float() / v_scale).clamp(-448, 448).to(e4m3)

        def call():
            paged_attention_triton(
                m["query"], kc_fp8, vc_fp8, out,
                m["query_start_loc"], m["seq_lens"], m["token_seq_idx"],
                m["block_table"], m["scale"], k_descale=k_scale, v_descale=v_scale,
                is_prefill=False)
    else:
        def call():
            paged_attention_triton(
                m["query"], m["key_cache"], m["value_cache"], out,
                m["query_start_loc"], m["seq_lens"], m["token_seq_idx"],
                m["block_table"], m["scale"], use_fp8=False, is_prefill=False)

    ms = _bench(call)

    # 每步读的 KV 字节：dense 读全部 ctx；fp8 半字节；sparse 只读 sparsity 比例。
    read_frac = 1.0
    bytes_per_elem = 2.0
    if path == "fp8KV":
        bytes_per_elem = 1.0
    if path == "sparse0.25":
        read_frac = float(os.environ.get("CUSTOM_SPARSITY", "0.25"))
    # K + V 两份，(bs 条 × ctx 位置 × NKV 头 × HS 维)
    kv_bytes = 2.0 * bs * ctx * NKV * HS * bytes_per_elem * read_frac
    tbps = kv_bytes / (ms * 1e-3) / 1e12
    tpot_ms = ms * NUM_LAYERS
    print("RESULT " + json.dumps(dict(ctx=ctx, bs=bs, path=path,
                                      ms=ms, tpot_ms=tpot_ms, tbps=tbps)))


# ---------------------------------------------------------------------------
# driver：spawn 子进程跑各配置，汇总
# ---------------------------------------------------------------------------
def driver(gpu, ctxs, bses, paths):
    import torch
    repo = str(Path(__file__).resolve().parents[1])
    dev_name = torch.cuda.get_device_name(gpu)
    print(f"device: cuda:{gpu} ({dev_name})  dtype=bfloat16  layers={NUM_LAYERS}")
    print(f"维度: num_heads={NH} num_kv_heads={NKV} head_size={HS} block_size={BS} (Qwen3-32B)\n")

    results = {}   # (ctx,bs,path) -> dict
    for ctx in ctxs:
        for bs in bses:
            for path in paths:
                env = dict(os.environ)
                env.update(PATH_ENV[path])
                env["PYTHONPATH"] = repo + ":" + env.get("PYTHONPATH", "")
                cmd = [sys.executable, __file__, "--worker",
                       "--ctx", str(ctx), "--bs", str(bs), "--path", path, "--gpu", str(gpu)]
                try:
                    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
                    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
                    if line is None:
                        err = (p.stderr.strip().splitlines() or ["<no stderr>"])[-1]
                        print(f"[warn] {path} ctx={ctx} bs={bs} 失败: {err}")
                        continue
                    results[(ctx, bs, path)] = json.loads(line[len("RESULT "):])
                except subprocess.TimeoutExpired:
                    print(f"[warn] {path} ctx={ctx} bs={bs} 超时")

    # ---- 打印对比表 ----
    for bs in bses:
        print(f"\n### decode bs={bs}  (TPOT_attn = ms/step × {NUM_LAYERS} 层)\n")
        header = "| ctx | 配置 | ms/step | TPOT_attn(ms) | KV带宽(TB/s) | vs baseline |"
        sep = "| --- | --- | ---: | ---: | ---: | ---: |"
        print(header); print(sep)
        for ctx in ctxs:
            base = results.get((ctx, bs, "baseline"))
            base_ms = base["ms"] if base else None
            for path in paths:
                r = results.get((ctx, bs, path))
                if r is None:
                    print(f"| {ctx} | {path} | - | - | - | - |")
                    continue
                spd = f"{base_ms / r['ms']:.2f}x" if base_ms else "-"
                print(f"| {ctx} | {path} | {r['ms']:.3f} | {r['tpot_ms']:.1f} "
                      f"| {r['tbps']:.2f} | {spd} |")
    print("\n注：TPOT_attn 只含 attention kernel；端到端 TPOT 还包含其余 63 层的 MLP/norm/launch 开销，")
    print("故实际端到端 TPOT 会更大，但**各配置的相对加速**可直接反映 decode attention 的优化收益。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--worker", action="store_true", help="内部：测单个配置点")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--path", type=str)
    ap.add_argument("--ctxs", type=str, default="8192,32768,98304")
    ap.add_argument("--bses", type=str, default="1,8")
    ap.add_argument("--paths", type=str, default=",".join(PATHS))
    args = ap.parse_args()

    if args.worker:
        worker(args.ctx, args.bs, args.path, args.gpu)
        return 0

    ctxs = [int(x) for x in args.ctxs.split(",") if x]
    bses = [int(x) for x in args.bses.split(",") if x]
    paths = [p for p in args.paths.split(",") if p]
    driver(args.gpu, ctxs, bses, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
