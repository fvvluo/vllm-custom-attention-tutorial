#!/usr/bin/env python3
"""V4 adapter synthetic correctness + fallback gate (Liu Xiaochen).

READ-ONLY harness for the V4 dispatch. Verifies, on the REAL packed KV-cache
layout the CUSTOM backend produces — `(num_blocks, Hkv, block_size, 2*hs)` sliced
into K/V halves (slot-stride 2*hs, value-view offset +hs) — that:

  A. support domain: can_use_v3_decode == True; adapter V3 (try_v3_decode) is
     bit-identical to a direct paged_decode_v3 call, and both match a PyTorch
     reference within 5e-3; output is written in place (same data_ptr).
  B. fallback domain: can_use_v3_decode == False with a sensible reason for
     prefill / mixed / non-bf16 / wrong head_dim / wrong block_size / misaligned
     cache — and NO GPU tensor is silently cloned to force a hit.
  C. repeated A->B->A: adapter results are stable and do not capture first tensors.

This does NOT touch kernel math; it only drives the adapter + V3 runner.

Run:
  CUDA_VISIBLE_DEVICES=<gpu> LIUXIAOCHEN_PAGED_DECODE_V3=1 /usr/bin/python -u \
    custom_backend/liuxiaochen_paged_decode/verify_v4_adapter.py
"""

import os
import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.liuxiaochen_paged_decode.runner_v3 import paged_decode_v3
from custom_backend.liuxiaochen_paged_decode import vllm_adapter_v4 as v4

QH, KVH, D, BS = 64, 8, 128, 16
STRICT = 5e-3
_results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def _ceil(a, b):
    return (a + b - 1) // b


def build_packed(seq_lens, device, seed=0, shuffle=True):
    """Replicate the CUSTOM backend cache: packed (num_blocks,Hkv,BS,2*D),
    split into K/V halves -> slot-stride 2*D, value offset +D. Fill via
    block_table so V3/reference read the same values."""
    torch.manual_seed(seed)
    kv, queries = [], []
    for sl in seq_lens:
        kv.append((torch.randn(sl, KVH, D, device=device, dtype=torch.bfloat16),
                   torch.randn(sl, KVH, D, device=device, dtype=torch.bfloat16)))
        queries.append(torch.randn(1, QH, D, device=device, dtype=torch.bfloat16))
    bps = [_ceil(sl, BS) for sl in seq_lens]
    total_blocks = sum(bps) + 3
    max_nb = max(bps)
    packed = torch.randn(total_blocks, KVH, BS, 2 * D, device=device, dtype=torch.bfloat16)
    key_cache, value_cache = packed.split(D, dim=-1)          # views: slot-stride 2*D
    block_table = torch.zeros(len(seq_lens), max_nb, device=device, dtype=torch.int32)
    phys = list(range(total_blocks))
    if shuffle:
        g = torch.Generator(device="cpu"); g.manual_seed(seed + 1)
        phys = torch.randperm(total_blocks, generator=g).tolist()
    nxt = 0
    for si, sl in enumerate(seq_lens):
        for lb in range(bps[si]):
            pb = phys[nxt]; nxt += 1
            block_table[si, lb] = pb
            for j in range(BS):
                tok = lb * BS + j
                if tok < sl:
                    key_cache[pb, :, j, :] = kv[si][0][tok]
                    value_cache[pb, :, j, :] = kv[si][1][tok]
    q_lens = [1] * len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    qsl = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
    qsl[1:] = torch.tensor(q_lens, device=device).cumsum(0)
    num_tokens = int(qsl[-1])
    query = torch.cat(queries, dim=0)
    tsi = torch.searchsorted(qsl[1:], torch.arange(num_tokens, device=device, dtype=torch.int32),
                             right=True).to(torch.int32)
    return dict(kv=kv, query=query, key_cache=key_cache, value_cache=value_cache,
                packed=packed, block_table=block_table, qsl=qsl, seq_lens=seq_lens_t,
                tsi=tsi, num_tokens=num_tokens)


def reference(kv, queries, scale):
    outs = []
    for (k, v), q in zip(kv, queries):
        sl = k.shape[0]
        out = torch.empty((1, QH, D), dtype=torch.float32, device=q.device)
        qf, kf, vf = q.float(), k.float(), v.float()
        for h in range(QH):
            kvh = h // (QH // KVH)
            s = (qf[0, h] * scale) @ kf[:sl, kvh].T
            w = torch.softmax(s, dim=-1)
            out[0, h] = w @ vf[:sl, kvh]
        outs.append(out)
    return torch.cat(outs, dim=0)


def adapter_call(d, scale, max_seq_len, out):
    """Drive try_v3_decode exactly as the backend does."""
    return v4.try_v3_decode(
        query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
        output_view=out, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
        token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
        max_seq_len=max_seq_len, num_heads=QH, num_kv_heads=KVH, head_size=D,
        num_actual_tokens=d["num_tokens"], max_query_len=1, causal=True,
    )


def support_case(name, seq_lens, dev, seed=0, shuffle=True):
    scale = 1.0 / (D ** 0.5)
    d = build_packed(seq_lens, dev, seed=seed, shuffle=shuffle)
    msl = max(seq_lens)
    ref = reference(d["kv"], [d["query"][i:i+1] for i in range(len(seq_lens))], scale)

    ok_gate, reason = v4.can_use_v3_decode(
        query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
        output_view=torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16),
        num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=d["num_tokens"],
        num_seqs=len(seq_lens), max_query_len=1, causal=True, scale=scale)
    check(f"[{name}] can_use_v3_decode==True", ok_gate, f"reason={reason}")

    # direct V3
    out_direct = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)
    paged_decode_v3(query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
                    output=out_direct, query_start_loc=d["qsl"], seq_lens=d["seq_lens"],
                    token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
                    split_size_tokens=256, max_seq_len=msl)
    # adapter V3
    out_adapter = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)
    op0 = out_adapter.data_ptr()
    used = adapter_call(d, scale, msl, out_adapter)
    torch.cuda.synchronize()
    check(f"[{name}] adapter dispatched to V3", used)
    check(f"[{name}] adapter V3 == direct V3 (bit-identical)",
          torch.equal(out_adapter, out_direct),
          f"max_abs={(out_adapter.float()-out_direct.float()).abs().max().item():.3e}")
    ma = (out_adapter.float() - ref).abs().max().item()
    check(f"[{name}] adapter V3 vs reference <=5e-3", ma <= STRICT, f"max_abs={ma:.3e}")
    check(f"[{name}] output written in place (data_ptr stable)", out_adapter.data_ptr() == op0)
    check(f"[{name}] output shape/dtype ok",
          tuple(out_adapter.shape) == (d["num_tokens"], QH, D) and out_adapter.dtype == torch.bfloat16)
    check(f"[{name}] no NaN/Inf", not torch.isnan(out_adapter).any() and not torch.isinf(out_adapter).any())
    return d, scale, msl


def fallback_case(name, *, query, key_cache, value_cache, num_heads, num_kv_heads,
                  head_size, num_actual_tokens, num_seqs, max_query_len, causal, scale,
                  expect_substr):
    out = torch.empty(max(num_actual_tokens, 1), num_heads, head_size,
                      device=query.device, dtype=torch.bfloat16)
    ok, reason = v4.can_use_v3_decode(
        query=query, key_cache=key_cache, value_cache=value_cache, output_view=out,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=head_size,
        num_actual_tokens=num_actual_tokens, num_seqs=num_seqs,
        max_query_len=max_query_len, causal=causal, scale=scale)
    check(f"[fallback:{name}] can_use_v3_decode==False", not ok, f"reason={reason}")
    check(f"[fallback:{name}] reason contains '{expect_substr}'", expect_substr in reason,
          f"reason={reason}")


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    print(f"flag LIUXIAOCHEN_PAGED_DECODE_V3={os.environ.get('LIUXIAOCHEN_PAGED_DECODE_V3','0')} "
          f"v3_enabled={v4.v3_enabled()}")
    if not v4.v3_enabled():
        print("NOTE: flag off -> adapter would fall back; set LIUXIAOCHEN_PAGED_DECODE_V3=1 for full test")

    # ---------- A. support domain ----------
    print("\n[A] support domain")
    support_case("ns1_sl128", [128], dev, seed=0)
    support_case("ns1_sl8192", [8192], dev, seed=1)
    d4, scale4, msl4 = support_case("ns4_mixed", [128, 512, 4096, 8192], dev, seed=2)

    # ---------- C. repeated A->B->A via adapter ----------
    print("\n[C] repeated A->B->A (adapter)")
    dA = build_packed([8192], dev, seed=7)
    dB = build_packed([8192], dev, seed=99)
    oA1 = torch.empty(1, QH, D, device=dev, dtype=torch.bfloat16)
    oB = torch.empty(1, QH, D, device=dev, dtype=torch.bfloat16)
    oA2 = torch.empty(1, QH, D, device=dev, dtype=torch.bfloat16)
    adapter_call(dA, 1.0/(D**0.5), 8192, oA1)
    adapter_call(dB, 1.0/(D**0.5), 8192, oB)
    adapter_call(dA, 1.0/(D**0.5), 8192, oA2)
    torch.cuda.synchronize()
    check("[C] A1==A2 via adapter (no first-tensor capture)", torch.equal(oA1, oA2))
    check("[C] A!=B via adapter", not torch.equal(oA1, oB),
          f"max_abs={(oA1.float()-oB.float()).abs().max().item():.3e}")

    # ---------- B. fallback domain ----------
    print("\n[B] fallback domain")
    base = build_packed([256], dev, seed=3)
    # prefill: max_query_len>1
    fallback_case("prefill", query=torch.randn(8, QH, D, device=dev, dtype=torch.bfloat16),
                  key_cache=base["key_cache"], value_cache=base["value_cache"],
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=8,
                  num_seqs=1, max_query_len=8, causal=True, scale=1.0/(D**0.5),
                  expect_substr="not_pure_decode")
    # mixed: num_tokens != num_seqs (max_query_len still >1)
    fallback_case("mixed", query=torch.randn(6, QH, D, device=dev, dtype=torch.bfloat16),
                  key_cache=base["key_cache"], value_cache=base["value_cache"],
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=6,
                  num_seqs=4, max_query_len=3, causal=True, scale=1.0/(D**0.5),
                  expect_substr="not_pure_decode")
    # non-bf16 query
    fallback_case("fp16_query", query=torch.randn(1, QH, D, device=dev, dtype=torch.float16),
                  key_cache=base["key_cache"], value_cache=base["value_cache"],
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=1,
                  num_seqs=1, max_query_len=1, causal=True, scale=1.0/(D**0.5),
                  expect_substr="query_dtype")
    # unsupported head_dim
    fallback_case("head_dim64", query=torch.randn(1, QH, 64, device=dev, dtype=torch.bfloat16),
                  key_cache=base["key_cache"], value_cache=base["value_cache"],
                  num_heads=QH, num_kv_heads=KVH, head_size=64, num_actual_tokens=1,
                  num_seqs=1, max_query_len=1, causal=True, scale=1.0/(D**0.5),
                  expect_substr="head_size")
    # unsupported block_size (build a bs=32 packed cache)
    bs32 = torch.randn(10, KVH, 32, 2 * D, device=dev, dtype=torch.bfloat16)
    kc32, vc32 = bs32.split(D, dim=-1)
    fallback_case("block_size32", query=torch.randn(1, QH, D, device=dev, dtype=torch.bfloat16),
                  key_cache=kc32, value_cache=vc32,
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=1,
                  num_seqs=1, max_query_len=1, causal=True, scale=1.0/(D**0.5),
                  expect_substr="block_size")
    # misaligned cache: make last-dim non-contiguous (stride(-1)!=1)
    mis = torch.randn(10, KVH, BS, D, 2, device=dev, dtype=torch.bfloat16)[..., 0]  # stride(-1)=2
    fallback_case("misaligned_stride", query=torch.randn(1, QH, D, device=dev, dtype=torch.bfloat16),
                  key_cache=mis, value_cache=mis,
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=1,
                  num_seqs=1, max_query_len=1, causal=True, scale=1.0/(D**0.5),
                  expect_substr="unaligned")
    # non-causal
    fallback_case("non_causal", query=torch.randn(1, QH, D, device=dev, dtype=torch.bfloat16),
                  key_cache=base["key_cache"], value_cache=base["value_cache"],
                  num_heads=QH, num_kv_heads=KVH, head_size=D, num_actual_tokens=1,
                  num_seqs=1, max_query_len=1, causal=False, scale=1.0/(D**0.5),
                  expect_substr="non_causal")

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\nSUMMARY: {len(_results)-n_fail}/{len(_results)} PASS, {n_fail} FAIL")
    print("V4_ADAPTER_TEST=" + ("PASS" if n_fail == 0 else "FAIL"))
    sys.exit(0 if n_fail == 0 else 2)


if __name__ == "__main__":
    main()
