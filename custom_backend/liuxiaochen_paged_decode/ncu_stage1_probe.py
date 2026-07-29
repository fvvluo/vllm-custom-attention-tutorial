#!/usr/bin/env python3
"""Minimal ncu harness: run ONE Stage-1 launch (V2 or V3) for Nsight profiling.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> ncu ... /usr/bin/python -m ...ncu_stage1_probe --impl v3 --seq-len 131072 --split 256

JIT compile + warmup happen BEFORE the profiled region; ncu targets the Stage-1
kernel by name (--kernel-name regex). We launch Stage-1 a few times so ncu can
pick a stable instance. Uses the bench builder (shuffled block_table).
"""

import argparse
import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.liuxiaochen_paged_decode.bench_paged_decode_v3 import build, QH, D
from custom_backend.liuxiaochen_paged_decode.runner_v2 import build_v2_runners
from custom_backend.liuxiaochen_paged_decode.runner_v3 import build_v3_runners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["v2", "v3"], required=True)
    ap.add_argument("--seq-len", type=int, default=131072)
    ap.add_argument("--num-seqs", type=int, default=1)
    ap.add_argument("--split", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--launches", type=int, default=3)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}")
    seq_lens = [a.seq_len] * a.num_seqs
    d = build(seq_lens, dev, seed=0, shuffle=True)
    scale = 1.0 / (D ** 0.5)
    out = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)

    builder = build_v2_runners if a.impl == "v2" else build_v3_runners
    run_s1, run_cmb, reset_lse, nsm, ws = builder(
        query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
        output=out, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
        token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
        split_size_tokens=a.split, max_seq_len=a.seq_len)

    # JIT + warmup OUTSIDE profiled region.
    reset_lse()
    for _ in range(5):
        run_s1()
    torch.cuda.synchronize()

    # profiled region: wrap ONLY Stage-1 launches with the CUDA profiler API so
    # `ncu --profile-from-start off` captures nothing from the (huge) build phase.
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(a.launches):
        run_s1()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"{a.impl} stage1 done: seq_len={a.seq_len} split={a.split} nsm={nsm} ws={ws}")


if __name__ == "__main__":
    main()
