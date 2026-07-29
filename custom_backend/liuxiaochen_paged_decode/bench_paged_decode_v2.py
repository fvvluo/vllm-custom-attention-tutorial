#!/usr/bin/env python3
"""V2 microbenchmark: tutorial Triton paged decode vs V2 paged warp-MMA decode.

Same query / paged K-V / block_table / metadata / scale for both. Decode-only
(q_len==1 per seq). Timing (CUDA events, same process, randomized order) excludes
JIT, allocation, block_table construction, reference, error reduction, and any
GPU->CPU sync (max_seq_len is passed in, not derived from seq_lens.cpu()).

Run:
  CUDA_VISIBLE_DEVICES=<gpu> /usr/bin/python -u \
    -m custom_backend.liuxiaochen_paged_decode.bench_paged_decode_v2 \
    --seq-len 131072 --num-seqs 1 --split 256 --warmup 10 --iters 100 --rounds 5
"""

import argparse
import statistics
import random

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.triton_attention import paged_attention_triton
from custom_backend.liuxiaochen_paged_decode import paged_decode_v2, workspace_bytes
from custom_backend.liuxiaochen_paged_decode.runner_v2 import _ceil_div

QH, KVH, D, BS = 64, 8, 128, 16


def build(seq_lens, device, seed, shuffle=True):
    torch.manual_seed(seed)
    kv_data = []
    queries = []
    for sl in seq_lens:
        kv_data.append((torch.randn(sl, KVH, D, device=device, dtype=torch.bfloat16),
                        torch.randn(sl, KVH, D, device=device, dtype=torch.bfloat16)))
        queries.append(torch.randn(1, QH, D, device=device, dtype=torch.bfloat16))
    blocks_per_seq = [_ceil_div(sl, BS) for sl in seq_lens]
    total_blocks = sum(blocks_per_seq) + 3
    max_nb = max(blocks_per_seq)
    key_cache = torch.randn(total_blocks, KVH, BS, D, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn(total_blocks, KVH, BS, D, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(len(seq_lens), max_nb, device=device, dtype=torch.int32)
    phys = list(range(total_blocks))
    if shuffle:
        g = torch.Generator(device="cpu"); g.manual_seed(seed + 1)
        phys = torch.randperm(total_blocks, generator=g).tolist()
    nxt = 0
    for si, sl in enumerate(seq_lens):
        for lb in range(blocks_per_seq[si]):
            pb = phys[nxt]; nxt += 1
            block_table[si, lb] = pb
            for j in range(BS):
                tok = lb * BS + j
                if tok < sl:
                    key_cache[pb, :, j, :] = kv_data[si][0][tok]
                    value_cache[pb, :, j, :] = kv_data[si][1][tok]
    q_lens = [1] * len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    qsl = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
    qsl[1:] = torch.tensor(q_lens, device=device).cumsum(0)
    num_tokens = int(qsl[-1])
    query = torch.cat(queries, dim=0)
    tsi = torch.searchsorted(qsl[1:], torch.arange(num_tokens, device=device, dtype=torch.int32), right=True).to(torch.int32)
    return dict(query=query, key_cache=key_cache, value_cache=value_cache,
                block_table=block_table, qsl=qsl, seq_lens=seq_lens_t, tsi=tsi,
                num_tokens=num_tokens, total_blocks=total_blocks)


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--num-seqs", type=int, default=1)
    ap.add_argument("--split", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tri-iters", type=int, default=0, help="override triton iters (0=same as --iters)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                    help="randomized physical block mapping (default)")
    ap.add_argument("--identity", dest="shuffle", action="store_false",
                    help="sequential physical block mapping")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}")
    seq_lens = [a.seq_len] * a.num_seqs
    max_seq_len = a.seq_len

    d = build(seq_lens, dev, a.seed, shuffle=a.shuffle)
    scale = 1.0 / (D ** 0.5)
    out_tri = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)
    out_v2 = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)

    def run_tri():
        paged_attention_triton(query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
                               output=out_tri, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
                               token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale)

    def run_v2():
        paged_decode_v2(query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
                        output=out_v2, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
                        token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
                        split_size_tokens=a.split, max_seq_len=max_seq_len)

    # Stage-split accessors (pre-compiled, pre-reset workspace; excludes lse.fill_
    # and JIT/alloc). Stage-1-only and combine-only launched separately.
    from custom_backend.liuxiaochen_paged_decode.runner_v2 import build_v2_runners
    run_s1, run_cmb, reset_lse, nsm_dbg, ws_dbg = build_v2_runners(
        query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
        output=out_v2, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
        token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
        split_size_tokens=a.split, max_seq_len=max_seq_len)

    # JIT + warmup once (excluded from timing)
    run_tri(); run_v2(); reset_lse(); run_s1(); run_cmb(); torch.cuda.synchronize()
    # correctness sanity
    ma = (out_v2.float() - out_tri.float()).abs().max().item()

    tri_iters = a.tri_iters if a.tri_iters > 0 else a.iters
    tri_s, v2_s, s1_s, cmb_s = [], [], [], []
    names = ["tri", "v2", "s1", "cmb"]
    reset_lse(); torch.cuda.synchronize()
    for _ in range(a.rounds):
        order = names[:]; random.shuffle(order)
        for n in order:
            if n == "tri":
                tri_s.append(time_fn(run_tri, a.warmup, tri_iters))
            elif n == "v2":
                v2_s.append(time_fn(run_v2, a.warmup, a.iters))
            elif n == "s1":
                s1_s.append(time_fn(run_s1, a.warmup, a.iters))
            else:
                cmb_s.append(time_fn(run_cmb, a.warmup, a.iters))

    def stats(v):
        vs = sorted(v)
        return statistics.median(vs), min(vs), max(vs), vs[int(0.9*(len(vs)-1))]
    tm, tmn, tmx, tp90 = stats(tri_s)
    vm, vmn, vmx, vp90 = stats(v2_s)
    s1m = statistics.median(sorted(s1_s))
    cmbm = statistics.median(sorted(cmb_s))

    kv_bytes = 2 * sum(seq_lens) * KVH * D * 2  # unique K+V of referenced tokens (K+V)
    ws = workspace_bytes(a.num_seqs, QH, _ceil_div(max_seq_len, a.split), D)
    v2_gbs = kv_bytes / (vm / 1000) / 1e9
    tps = d["num_tokens"] / (vm / 1000)
    print(f"cfg: seq_len={a.seq_len} num_seqs={a.num_seqs} split={a.split} nsm={nsm_dbg} "
          f"shuffle={a.shuffle} warmup={a.warmup} iters={a.iters} tri_iters={tri_iters} rounds={a.rounds}")
    print(f"  sanity max_abs(v2 vs tri)={ma:.3e}")
    print(f"  tutorial-triton : median={tm:.4f} min={tmn:.4f} max={tmx:.4f} p90={tp90:.4f} ms")
    print(f"  V2 Stage-1-only : median={s1m:.4f} ms")
    print(f"  V2 combine-only : median={cmbm:.4f} ms")
    print(f"  V2 end-to-end   : median={vm:.4f} min={vmn:.4f} max={vmx:.4f} p90={vp90:.4f} ms")
    print(f"  (note: Stage-1 + combine != end-to-end exactly; each has own launch/event/lse-reset overhead)")
    print(f"  V2 speedup vs triton={tm/vm:.2f}x | unique-KV(K+V) GB/s={v2_gbs:.0f} | tokens/s={tps:.0f} | workspace={ws} B")



if __name__ == "__main__":
    main()
