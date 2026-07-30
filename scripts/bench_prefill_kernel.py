"""Prefill kernel 微基准：bf16 vs 动态量化 fp8 vs 预量化 fp8 KV。

直接调 paged_attention_triton（不起服务，快速可靠），造合成分页 KV，测 prefill TTFT-代表
（单次前向时间）。回答：fp8 到底能不能加速 prefill，哪种 fp8 路径最快。
"""
import argparse
import sys
import time
from pathlib import Path

import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from custom_backend.triton_attention import paged_attention_triton  # noqa: E402


def make_paged(num_tokens, num_heads, num_kv_heads, head_size, block_size, dtype, dev, fp8=False):
    q = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=dev)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kc_dt = torch.float8_e4m3fn if fp8 else dtype
    kc = torch.randn(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=dev)
    vc = torch.randn(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=dev)
    if fp8:
        kc = (kc / 4.0).to(torch.float8_e4m3fn)
        vc = (vc / 4.0).to(torch.float8_e4m3fn)
    out = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=dev)
    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=dev)
    tsi = torch.zeros(num_tokens, dtype=torch.int32, device=dev)
    bt = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, num_blocks)
    return q, kc, vc, out, qsl, seq_lens, tsi, bt


def bench(fn, warmup, iters, dev):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize(dev)
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--num-heads", type=int, default=64)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    dtype = torch.bfloat16
    scale = 1.0 / (args.head_size ** 0.5)

    print(f"Prefill 微基准  heads={args.num_heads}/{args.num_kv_heads} d={args.head_size} block={args.block_size}")
    print(f"{'seq_len':>8}{'bf16(ms)':>11}{'fp8-dyn(ms)':>13}{'fp8-prequant(ms)':>18}{'dyn vs bf16':>13}{'prequant vs bf16':>18}")
    print("-" * 82)
    for n in args.seqs:
        # bf16 KV
        q, kc, vc, out, qsl, sl, tsi, bt = make_paged(n, args.num_heads, args.num_kv_heads,
                                                      args.head_size, args.block_size, dtype, dev, fp8=False)
        t_bf = bench(lambda: paged_attention_triton(q, kc, vc, out, qsl, sl, tsi, bt, scale,
                                                    use_fp8=False, is_prefill=True), args.warmup, args.iters, dev)
        t_dyn = bench(lambda: paged_attention_triton(q, kc, vc, out, qsl, sl, tsi, bt, scale,
                                                     use_fp8=True, is_prefill=True), args.warmup, args.iters, dev)
        # 预量化 fp8 KV
        qf, kcf, vcf, outf, qslf, slf, tsif, btf = make_paged(n, args.num_heads, args.num_kv_heads,
                                                              args.head_size, args.block_size, dtype, dev, fp8=True)
        kdesc = torch.ones(1, dtype=torch.float32, device=dev) * 4.0
        vdesc = torch.ones(1, dtype=torch.float32, device=dev) * 4.0
        t_pre = bench(lambda: paged_attention_triton(qf, kcf, vcf, outf, qslf, slf, tsif, btf, scale,
                                                     use_fp8=True, k_descale=kdesc, v_descale=vdesc,
                                                     is_prefill=True), args.warmup, args.iters, dev)
        print(f"{n:>8}{t_bf:>11.3f}{t_dyn:>13.3f}{t_pre:>18.3f}"
              f"{t_bf/t_dyn:>12.2f}x{t_bf/t_pre:>17.2f}x")


if __name__ == "__main__":
    main()
