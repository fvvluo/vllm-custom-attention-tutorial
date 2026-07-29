# SPDX-License-Identifier: Apache-2.0
"""
FP8 分页 attention kernel 性能微基准（kernel-level，脱离 vLLM 服务）
=====================================================================

直接对 custom_backend.triton_attention.paged_attention_triton 计时，对比：
  - fp8 路径（use_fp8=True，本仓库交付）
  - bf16 路径（use_fp8=False，同一 kernel 的对照分支）
  - 教程原始朴素 kernel（若可 import，作为 baseline）

覆盖 Qwen3-32B 维度（num_heads=64, num_kv_heads=8, head_size=128, block_size=16），
分别测 prefill（单条请求整段 query）与 decode（已有 context、本步 1 token）。

指标：kernel 平均耗时(ms)、等效 attention FLOPs 吞吐(TFLOP/s)。注意本教学 kernel 的 grid 是
(num_tokens × num_heads)、每 program 串行扫 KV，FLOPs 吞吐会远低于专用 FA3，仅作 fp8 vs bf16
的相对对比与量级参考。

运行：
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python scripts/bench_paged_kernel.py --gpu 1
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_backend.triton_attention import paged_attention_triton  # noqa: E402

try:
    # 教程原始朴素 kernel（git 里已被替换；若本地留有备份则可对照）
    from custom_backend._orig_triton_attention import (  # type: ignore  # noqa: E402
        paged_attention_triton as naive_paged_attention,
    )
    _HAS_NAIVE = True
except Exception:  # noqa: BLE001
    _HAS_NAIVE = False


def build_inputs(seqs, num_heads, num_kv_heads, head_size, block_size, dev, dtype):
    """seqs: list[(seq_len, query_len)] -> 分页 KV cache + 元数据（同教程测试布局）。"""
    torch.manual_seed(0)
    kv, qs = [], []
    for (sl, ql) in seqs:
        k = torch.randn(sl, num_kv_heads, head_size, device=dev, dtype=dtype)
        v = torch.randn(sl, num_kv_heads, head_size, device=dev, dtype=dtype)
        q = torch.randn(ql, num_heads, head_size, device=dev, dtype=dtype)
        kv.append((k, v)); qs.append(q)

    blocks_per_seq = [(len(k) + block_size - 1) // block_size for (k, _) in kv]
    total_blocks = sum(blocks_per_seq) + 2
    max_num_blocks = max(blocks_per_seq)
    key_cache = torch.zeros((total_blocks, num_kv_heads, block_size, head_size),
                            device=dev, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((len(kv), max_num_blocks), device=dev, dtype=torch.int32)
    nb = 0
    for si, (k, v) in enumerate(kv):
        for j in range(len(k)):
            lb, slot = j // block_size, j % block_size
            if slot == 0:
                block_table[si, lb] = nb; nb += 1
            pb = block_table[si, lb]
            key_cache[pb, :, slot, :] = k[j]
            value_cache[pb, :, slot, :] = v[j]

    q_lens = [q.shape[0] for q in qs]
    qsl = torch.zeros(len(seqs) + 1, device=dev, dtype=torch.int32)
    qsl[1:] = torch.tensor(q_lens, device=dev).cumsum(0)
    num_tokens = int(qsl[-1])
    query = torch.cat(qs, 0)
    seq_lens = torch.tensor([s for s, _ in seqs], device=dev, dtype=torch.int32)
    tsi = torch.searchsorted(qsl[1:], torch.arange(num_tokens, device=dev, dtype=torch.int32),
                             right=True).to(torch.int32)
    scale = 1.0 / head_size ** 0.5
    return dict(query=query, key_cache=key_cache, value_cache=value_cache,
                query_start_loc=qsl, seq_lens=seq_lens, token_seq_idx=tsi,
                block_table=block_table, scale=scale, num_tokens=num_tokens,
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=head_size)


def attn_flops(meta, seqs):
    """等效 attention FLOPs：每 (query token, head) 对其 causal 可见 KV 做 QK^T + PV。"""
    nh, hs = meta["num_heads"], meta["head_size"]
    total = 0.0
    for (sl, ql) in seqs:
        ctx = sl - ql
        for i in range(ql):
            visible = ctx + i + 1
            total += 2.0 * 2.0 * nh * visible * hs  # QK^T + PV
    return total


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters  # ms


def run(name, seqs, dev, dtype):
    nh, nkv, hs, bs = 64, 8, 128, 16
    m = build_inputs(seqs, nh, nkv, hs, bs, dev, dtype)
    out = torch.empty(m["num_tokens"], nh, hs, device=dev, dtype=dtype)
    flops = attn_flops(m, seqs)

    def call(use_fp8):
        paged_attention_triton(
            m["query"], m["key_cache"], m["value_cache"], out,
            m["query_start_loc"], m["seq_lens"], m["token_seq_idx"],
            m["block_table"], m["scale"], use_fp8=use_fp8)

    # 预量化 fp8 KV cache（e4m3 常驻，每字节减半）：模拟 --kv-cache-dtype fp8
    kc, vc = m["key_cache"], m["value_cache"]
    e4m3 = torch.float8_e4m3fn
    k_scale = (kc.abs().amax().clamp_min(1e-12) / 448.0).float().view(1)
    v_scale = (vc.abs().amax().clamp_min(1e-12) / 448.0).float().view(1)
    kc_fp8 = (kc.float() / k_scale).clamp(-448, 448).to(e4m3)
    vc_fp8 = (vc.float() / v_scale).clamp(-448, 448).to(e4m3)

    def call_pq():
        paged_attention_triton(
            m["query"], kc_fp8, vc_fp8, out,
            m["query_start_loc"], m["seq_lens"], m["token_seq_idx"],
            m["block_table"], m["scale"], k_descale=k_scale, v_descale=v_scale)

    t_fp8 = bench(lambda: call(True))
    t_bf16 = bench(lambda: call(False))
    t_pq = bench(call_pq)
    print(f"[{name}] num_tokens={m['num_tokens']:>6}")
    print(f"    bf16            : {t_bf16:8.3f} ms   {flops / t_bf16 / 1e9:7.2f} GFLOP/s")
    print(f"    fp8(动态量化KV) : {t_fp8:8.3f} ms   (相对 bf16: {t_bf16 / t_fp8:.2f}x)")
    print(f"    fp8(预量化KV常驻): {t_pq:8.3f} ms   (相对 bf16: {t_bf16 / t_pq:.2f}x)"
          f"  <- KV 半字节、不重量化")
    if _HAS_NAIVE:
        def call_naive():
            naive_paged_attention(
                m["query"], m["key_cache"], m["value_cache"], out,
                m["query_start_loc"], m["seq_lens"], m["token_seq_idx"],
                m["block_table"], m["scale"])
        t_naive = bench(call_naive)
        print(f"    教程朴素: {t_naive:8.3f} ms   (fp8 相对朴素: {t_naive / t_fp8:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    dev = f"cuda:{args.gpu}"; dtype = torch.bfloat16
    print(f"device: {dev} ({torch.cuda.get_device_name(args.gpu)})  dtype={dtype}")
    print("维度: num_heads=64 num_kv_heads=8 head_size=128 block_size=16 (Qwen3-32B)\n")

    print("== PREFILL：单条请求整段 query ==")
    for s in (512, 2048, 8192, 32768):
        run(f"prefill s={s}", [(s, s)], dev, dtype)

    print("\n== DECODE：已有 context，本步 1 token（batch 多条）==")
    for ctx in (8192, 32768):
        run(f"decode ctx={ctx} bs=8", [(ctx, 1)] * 8, dev, dtype)


if __name__ == "__main__":
    main()
