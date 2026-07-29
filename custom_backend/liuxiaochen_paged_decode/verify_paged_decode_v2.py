#!/usr/bin/env python3
"""V2 paged-decode correctness: V2 vs tutorial PyTorch reference vs tutorial Triton.

Builds an independent randomized paged KV cache (optionally shuffled block_table,
padded strides), runs all three on identical inputs. Decode-only (q_len==1).

Run:
  CUDA_VISIBLE_DEVICES=<gpu> /usr/bin/python -u \
    -m custom_backend.liuxiaochen_paged_decode.verify_paged_decode_v2 [--case ...]
Rule: tutorial tol rtol=atol=2e-2; also report strict max_abs<=5e-3 diagnostic.
"""

import argparse
import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.triton_attention import paged_attention_triton
from custom_backend.liuxiaochen_paged_decode import paged_decode_v2, workspace_bytes

QH, KVH, G, D, BS = 64, 8, 8, 128, 16
DT = torch.bfloat16
RTOL = ATOL = 2e-2
STRICT = 5e-3


def build_paged(seq_lens, device, dtype, shuffle=False, pad_lead=False, seed=0):
    """Build paged K/V cache + block_table + queries for decode-only (q_len=1 each)."""
    torch.manual_seed(seed)
    kv_data = []
    queries = []
    for sl in seq_lens:
        k = torch.randn(sl, KVH, D, device=device, dtype=dtype)
        v = torch.randn(sl, KVH, D, device=device, dtype=dtype)
        q = torch.randn(1, QH, D, device=device, dtype=dtype)  # decode: 1 token
        kv_data.append((k, v)); queries.append(q)
    blocks_per_seq = [ (sl + BS - 1)//BS for sl in seq_lens ]
    total_blocks = sum(blocks_per_seq) + 3
    max_num_blocks = max(blocks_per_seq)

    if pad_lead:
        # larger backing tensor sliced -> non-default leading stride, same logical shape
        backing_k = torch.randn(total_blocks, KVH, BS, D + 8, device=device, dtype=dtype)
        backing_v = torch.randn(total_blocks, KVH, BS, D + 8, device=device, dtype=dtype)
        key_cache = backing_k[..., :D]
        value_cache = backing_v[..., :D]
    else:
        # fill unreferenced blocks with distinct random noise so a wrong physical
        # read is numerically obvious
        key_cache = torch.randn(total_blocks, KVH, BS, D, device=device, dtype=dtype)
        value_cache = torch.randn(total_blocks, KVH, BS, D, device=device, dtype=dtype)

    block_table = torch.zeros(len(seq_lens), max_num_blocks, device=device, dtype=torch.int32)
    # assign physical blocks (optionally a shuffled permutation)
    phys = list(range(total_blocks))
    if shuffle:
        g = torch.Generator(device="cpu"); g.manual_seed(seed + 777)
        perm = torch.randperm(total_blocks, generator=g).tolist()
        phys = perm
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
    return kv_data, queries, key_cache, value_cache, block_table


def naive_reference(kv_data, queries, scale):
    outs = []
    for (k, v), q in zip(kv_data, queries):
        q_len, num_heads, hs = q.shape
        seq_len = k.shape[0]
        context_len = seq_len - q_len
        group = num_heads // KVH
        out = torch.empty((q_len, num_heads, hs), dtype=torch.float32, device=q.device)
        qf, kf, vf = q.float(), k.float(), v.float()
        for h in range(num_heads):
            kvh = h // group
            for i in range(q_len):
                abs_pos = context_len + i
                scores = (qf[i, h] * scale) @ kf[:abs_pos + 1, kvh].T
                w = torch.softmax(scores, dim=-1)
                out[i, h] = w @ vf[:abs_pos + 1, kvh]
        outs.append(out)
    return outs


def build_metadata(seq_lens, queries, device):
    q_lens = [q.shape[0] for q in queries]
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    qsl = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
    qsl[1:] = torch.tensor(q_lens, device=device).cumsum(0)
    num_tokens = int(qsl[-1])
    query = torch.cat(queries, dim=0)
    tsi = torch.searchsorted(qsl[1:], torch.arange(num_tokens, device=device, dtype=torch.int32), right=True).to(torch.int32)
    return query, qsl, seq_lens_t, tsi, num_tokens


def run_case(name, seq_lens, device, scale=None, shuffle=False, pad_lead=False,
             split_size=256, seed=0, strict_diag=True):
    kv_data, queries, key_cache, value_cache, block_table = build_paged(
        seq_lens, device, DT, shuffle=shuffle, pad_lead=pad_lead, seed=seed)
    query, qsl, seq_lens_t, tsi, num_tokens = build_metadata(seq_lens, queries, device)
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    max_seq_len = max(seq_lens)

    ref = torch.cat(naive_reference(kv_data, queries, scale), dim=0)  # [num_tokens, QH, D] fp32

    # tutorial triton
    out_tri = torch.empty(num_tokens, QH, D, device=device, dtype=DT)
    paged_attention_triton(query=query, key_cache=key_cache, value_cache=value_cache,
                           output=out_tri, query_start_loc=qsl, seq_lens=seq_lens_t,
                           token_seq_idx=tsi, block_table=block_table, scale=scale)

    # V2
    out_v2 = torch.empty(num_tokens, QH, D, device=device, dtype=DT)
    qp, kp, vp, op = query.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(), out_v2.data_ptr()
    mem0 = torch.cuda.memory_allocated(device)
    paged_decode_v2(query=query, key_cache=key_cache, value_cache=value_cache,
                    output=out_v2, query_start_loc=qsl, seq_lens=seq_lens_t,
                    token_seq_idx=tsi, block_table=block_table, scale=scale,
                    split_size_tokens=split_size, max_seq_len=max_seq_len)
    mem1 = torch.cuda.memory_allocated(device)
    same_ptr = (query.data_ptr() == qp and key_cache.data_ptr() == kp and
                value_cache.data_ptr() == vp and out_v2.data_ptr() == op)

    v2f = out_v2.float(); trif = out_tri.float()
    ns = len(seq_lens); mnb = int(block_table.shape[1])
    nsm = (max_seq_len + split_size - 1) // split_size
    def stats(a, b):
        ad = (a - b).abs(); return ad.max().item(), (ad / (b.abs() + 1e-6)).max().item()
    ma_ref, mr_ref = stats(v2f, ref)
    ma_tri, mr_tri = stats(v2f, trif)
    nan = bool(torch.isnan(v2f).any()); inf = bool(torch.isinf(v2f).any())
    tol_ok = torch.allclose(v2f, ref, rtol=RTOL, atol=ATOL) and (not nan) and (not inf)
    strict_ok = ma_ref <= STRICT
    print(f"[{name}] seq_lens={seq_lens} num_seqs={ns} num_tokens={num_tokens} "
          f"num_blocks={key_cache.shape[0]} bt={tuple(block_table.shape)}/{block_table.stride()} "
          f"kc={tuple(key_cache.shape)}/{key_cache.stride()} split={split_size} nsm={nsm}")
    print(f"    vs REF: max_abs={ma_ref:.3e} max_rel={mr_ref:.3e} | vs TRITON: max_abs={ma_tri:.3e} "
          f"| nan={nan} inf={inf} same_ptr={same_ptr} alloc_delta={mem1-mem0}B "
          f"ws={workspace_bytes(ns, QH, nsm, D)}B")
    print(f"    tutorial_tol(2e-2)={'PASS' if tol_ok else 'FAIL'}  strict(5e-3 vs ref)={'PASS' if strict_ok else 'WARN'}")
    return tol_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="basic",
                    choices=["basic", "tutorial", "irregular", "shuffle", "stride", "scale", "big", "all"])
    ap.add_argument("--split", type=int, default=256)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    ok = True

    if a.suite in ("basic", "all"):
        for sl in [1, 15, 16, 17, 63, 64, 65, 128, 512]:
            for seed in [0, 1, 2026]:
                ok &= run_case(f"basic sl={sl} seed={seed}", [sl], dev, split_size=a.split, seed=seed)
                if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite in ("tutorial", "all"):
        for seed in [0, 1, 2026]:
            ok &= run_case(f"tutorial-decode seed={seed}", [40, 17, 128], dev, split_size=a.split, seed=seed)
            if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite in ("irregular", "all"):
        for sls in [[1, 17, 65, 257], [127, 128, 129, 1023]]:
            ok &= run_case(f"irregular {sls}", sls, dev, split_size=a.split)
            if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite in ("shuffle", "all"):
        for seed in [0, 2026]:
            ok &= run_case(f"shuffle-bt seed={seed}", [127, 128, 129, 1023], dev,
                           shuffle=True, split_size=a.split, seed=seed)
            if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite in ("stride", "all"):
        ok &= run_case("padded-lead-stride", [65, 130, 257], dev, pad_lead=True, split_size=a.split)
        if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite in ("scale", "all"):
        ok &= run_case("scale-default", [128, 257], dev, scale=1.0/(D**0.5), split_size=a.split)
        ok &= run_case("scale-explicit", [128, 257], dev, scale=0.05, split_size=a.split)
        if not ok: print("FAIL — stop"); sys.exit(2)
    if a.suite == "big":
        ok &= run_case("8k", [40, 512, 8192], dev, split_size=a.split, shuffle=True)
        ok &= run_case("128k-single", [131072], dev, split_size=a.split, shuffle=True)
        if not ok: print("FAIL — stop"); sys.exit(2)
        ok &= run_case("128k-multi", [8192, 32768, 131072], dev, split_size=a.split, shuffle=True)
        if not ok: print("FAIL — stop"); sys.exit(2)

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
