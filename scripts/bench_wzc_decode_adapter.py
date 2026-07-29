#!/usr/bin/env python
"""
bench_wzc_decode_adapter.py — measure where decode time actually goes.
=====================================================================

Goal: let you verify *yourself* whether the ~9.8 tok/s decode throughput of the
CUSTOM backend is caused by the attention kernel/adapter, or by model-wide eager
dispatch (hundreds of kernel launches per decode step across the 32B model, not
captured into a CUDA graph).

It measures, on the target GQA family (q_heads=64, kv_heads=8, head_dim=128, bf16):
  (A) raw wzc paged decode kernel latency (isolated, kernel-native zero-copy pool),
  (B) full adapter call `paged_attention_wzc` for one decode token (kernel + the
      adapter's per-call host work: query_start_loc.tolist()/seq_lens.tolist() +
      the python per-request loop),
  (C) adapter host overhead = (B) - (A).

Then compare against the whole-model per-token budget:
  per-token time = 1 / decode_tps.  At the measured 9.8 tok/s that is ~102 ms.
If (B) << 102 ms, the attention layer is a tiny fraction and the ~102 ms/token is
dominated by the rest of the 32B model running eager (un-graphed) — which is what
`--enforce-eager` + the backend's `_cudagraph_support = NEVER` forces (the whole
model degrades to per-kernel launch, not just attention).

You said "a CUDA kernel launch is only a few us" — correct for ONE launch. The
point of this script is to show it is *hundreds of launches per step x tens of
thousands of steps*, and that graph replay (baseline) collapses that to one replay.

Run (pick a free GPU):
  cd /dockerdata/wangzicheng/vllm-custom-attention-tutorial
  CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=/dockerdata/landojiang/vllm_src:. \
      python3 scripts/bench_wzc_decode_adapter.py --seq 4096 32768 131072

Optionally estimate the model-wide launch cost with --launch-probe: it times N
trivial kernel launches on an empty-graph vs graph-replay to show the per-launch
dispatch cost your GPU actually pays in eager mode.
"""
import argparse
import math
import os
import sys

import torch

# Resolve `custom_backend` / `tests` to THIS repo (wangzicheng), overriding any
# system-wide editable install (e.g. one that maps them to /dockerdata/liujinrui).
# Put this repo's root at the FRONT of sys.path so it wins over site-packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)
# The wzc kernels live in the attention-test ops package.
_ATTN_TEST_DIR = os.environ.get(
    "WZC_ATTENTION_TEST_DIR", "/dockerdata/wangzicheng/attention-test")
if _ATTN_TEST_DIR not in sys.path:
    sys.path.insert(1, _ATTN_TEST_DIR)


def _bench(fn, it=50, wu=15):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it   # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="*", default=[4096, 32768, 131072])
    ap.add_argument("--q-heads", type=int, default=64)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--decode-tps", type=float, default=9.8,
                    help="measured CUSTOM decode throughput (tok/s) for the "
                         "per-token budget comparison; baseline flash_attn ~44")
    ap.add_argument("--launch-probe", action="store_true",
                    help="also probe raw per-kernel-launch dispatch cost "
                         "(eager loop vs one CUDA-graph replay)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need GPU"
    dev = "cuda"
    H, HK, hd = args.q_heads, args.kv_heads, args.head_dim
    P = 128  # kernel page size
    s = 1.0 / math.sqrt(hd)

    from ops import _wzc_paged_attn_decode as D
    from custom_backend import wzc_sparse_attention as A
    from tests.test_paged_attn_correctness import make_paged_kv_cache

    print(f"{'seq':>8} {'(A)kernel_us':>13} {'(B)adapter_us':>14} "
          f"{'(C)host_ovh_us':>15} {'per-tok_budget_ms':>18} {'attn_%_of_tok':>14}")
    budget_ms = 1000.0 / args.decode_tps
    for S in args.seq:
        npag = (S + P - 1) // P
        # (A) raw kernel with a kernel-native pool (zero-copy, no repack).
        kc = torch.randn(npag, HK, P, hd, device=dev, dtype=torch.bfloat16)
        vc = torch.randn(npag, HK, P, hd, device=dev, dtype=torch.bfloat16)
        bt = torch.arange(npag, dtype=torch.int32, device=dev).view(1, npag)
        sl = torch.tensor([S], dtype=torch.int32, device=dev)
        q = torch.randn(H, hd, device=dev, dtype=torch.bfloat16)
        D.PagedKVDecoder.decode(q, kc, vc, bt, sl, 0, sm_scale=s)
        torch.cuda.synchronize()
        ms_a = _bench(lambda: D.PagedKVDecoder.decode(q, kc, vc, bt, sl, 0, sm_scale=s))

        # (B) full adapter decode call (block_size=128 -> zero-copy path).
        k = torch.randn(S, HK, hd, device=dev, dtype=torch.bfloat16)
        v = torch.randn(S, HK, hd, device=dev, dtype=torch.bfloat16)
        q3 = torch.randn(1, H, hd, device=dev, dtype=torch.bfloat16)
        kc2, vc2, bt2 = make_paged_kv_cache([(k, v)], HK, hd, P, dev, torch.bfloat16)
        sl2 = torch.tensor([S], device=dev, dtype=torch.int32)
        qsl = torch.tensor([0, 1], device=dev, dtype=torch.int32)
        tsi = torch.zeros(1, device=dev, dtype=torch.int32)
        out = torch.empty(1, H, hd, device=dev, dtype=torch.bfloat16)
        call = lambda: A.paged_attention_wzc(
            query=q3, key_cache=kc2, value_cache=vc2, output=out,
            query_start_loc=qsl, seq_lens=sl2, token_seq_idx=tsi,
            block_table=bt2, scale=s)
        call()
        torch.cuda.synchronize()
        ms_b = _bench(call)

        host_ovh_us = (ms_b - ms_a) * 1000
        attn_pct = ms_b / budget_ms * 100
        print(f"{S:>8} {ms_a*1000:>13.1f} {ms_b*1000:>14.1f} "
              f"{host_ovh_us:>15.1f} {budget_ms:>18.1f} {attn_pct:>13.2f}%")

    if args.launch_probe:
        print("\n[launch-probe] per-kernel-launch dispatch cost "
              "(eager loop vs one graph replay):")
        x = torch.zeros(1024, device=dev)
        N = 400  # ~ launches per decode step for a deep model
        def eager_many():
            for _ in range(N):
                x.add_(1.0)
        ms_eager = _bench(eager_many, it=20, wu=5)
        g = torch.cuda.CUDAGraph()
        eager_many()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(N):
                x.add_(1.0)
        ms_graph = _bench(lambda: g.replay(), it=20, wu=5)
        print(f"  {N} eager launches : {ms_eager*1000:8.1f} us "
              f"({ms_eager*1000/N:.2f} us/launch)")
        print(f"  {N} graph replay   : {ms_graph*1000:8.1f} us "
              f"({ms_graph*1000/N:.2f} us/launch)")
        print(f"  => graph eliminates ~{(ms_eager-ms_graph)*1000:.0f} us of "
              f"dispatch per {N}-launch step. A 32B decode step has hundreds of\n"
              f"     such launches; x tens of thousands of steps = the minutes you saw.")

    print("\nInterpretation:")
    print("  If (B) adapter_us << per-token budget, attention is NOT the bottleneck;")
    print("  the per-token time is dominated by model-wide EAGER dispatch that a")
    print("  CUDA graph (baseline) would replay in one shot. Fix = make the backend")
    print("  CUDA-graph-capturable (batched decode kernel + _cudagraph_support) and")
    print("  drop --enforce-eager, NOT further tuning the attention kernel.")


if __name__ == "__main__":
    main()
