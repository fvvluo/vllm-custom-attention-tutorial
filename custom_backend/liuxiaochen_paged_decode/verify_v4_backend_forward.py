#!/usr/bin/env python3
"""V4 real backend-forward gate (Liu Xiaochen).

Exercises the REAL `CustomTritonImpl.forward` path — real metadata, real KV-cache
WRITE (`triton_reshape_and_cache_flash`), real read dispatch (V3 adapter or Triton
fallback) — WITHOUT loading Qwen3-32B. This is stronger than verify_v4_adapter.py,
which drove `try_v3_decode` directly.

REAL vs FIXTURE boundary:
  REAL:   CustomTritonImpl (normal __init__), CustomTritonMetadata, the forward()
          method itself, the packed KV-cache `(num_blocks,Hkv,BS,2*hs)`, block_table,
          slot_mapping, seq_lens, query_start_loc, max_query_len/max_seq_len, the
          KV-cache write, the V3-vs-Triton dispatch, and the output buffer contract
          `[num_tokens, num_heads*head_size]`.
  FIXTURE: a tiny `_Layer` object exposing only `_k_scale`/`_v_scale` (= 1.0), which
          is all `forward` reads from `layer`; and hand-built paged tensors standing
          in for what vLLM's model runner would supply. No vLLM engine/model/runner.

Decode cases pre-fill context slots 0..seq_len-2 in the cache and let forward WRITE
the last (decode) token, so a broken write would corrupt the result. Reference is a
plain PyTorch attention over the full per-sequence K/V.

Run:
  CUDA_VISIBLE_DEVICES=<gpu> LIUXIAOCHEN_PAGED_DECODE_V3=<0|1> /usr/bin/python -u \
    custom_backend/liuxiaochen_paged_decode/verify_v4_backend_forward.py
"""

import os
import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.custom_triton_backend import CustomTritonImpl, CustomTritonMetadata
from custom_backend.liuxiaochen_paged_decode import vllm_adapter_v4 as v4
from custom_backend.liuxiaochen_paged_decode.runner_v3 import paged_decode_v3

QH, KVH, D, BS = 64, 8, 128, 16
# STRICT (5e-3): used for the V3 decode path, whose fp32-accumulate MMA is very close
# to the fp32 reference. TRITON_TOL (2e-2, the tutorial's standard bf16 tolerance,
# rtol=atol=2e-2): used for the Triton FALLBACK path (prefill / fp16), whose bf16
# online-softmax accumulation legitimately diverges more from an fp32 reference. Using
# 5e-3 there would be an over-tight fixture threshold, not a real correctness failure.
STRICT = 5e-3
TRITON_TOL = 2e-2
_results = []


class _Layer:
    """FIXTURE: forward() only reads _k_scale/_v_scale from `layer`."""
    def __init__(self, device):
        self._k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)


def check(name, ok, detail=""):
    ok = bool(ok)
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def _ceil(a, b):
    return (a + b - 1) // b


def build_decode(seq_lens, device, seed, dtype=torch.bfloat16, shuffle=True, prefill_last=False):
    """Build a REAL packed paged cache + metadata for a decode batch (q_len==1 each).

    Context positions 0..seq_len-2 are written into the cache directly; the decode
    token (position seq_len-1) is returned as `key`/`value` for forward to WRITE via
    reshape_and_cache. `prefill_last=True` also pre-writes the last slot (so the test
    still works if we want the read-only view populated), but by default we leave it
    to forward's write path.
    """
    torch.manual_seed(seed)
    kv, queries = [], []
    for sl in seq_lens:
        kv.append((torch.randn(sl, KVH, D, device=device, dtype=dtype),
                   torch.randn(sl, KVH, D, device=device, dtype=dtype)))
        queries.append(torch.randn(1, QH, D, device=device, dtype=dtype))
    bps = [_ceil(sl, BS) for sl in seq_lens]
    total_blocks = sum(bps) + 3
    max_nb = max(bps)
    packed = torch.randn(total_blocks, KVH, BS, 2 * D, device=device, dtype=dtype)
    key_cache, value_cache = packed.split(D, dim=-1)
    block_table = torch.zeros(len(seq_lens), max_nb, device=device, dtype=torch.int32)
    phys = list(range(total_blocks))
    if shuffle:
        g = torch.Generator(device="cpu"); g.manual_seed(seed + 1)
        phys = torch.randperm(total_blocks, generator=g).tolist()
    nxt = 0
    slot_map = []
    cur_k = torch.empty(len(seq_lens), KVH, D, device=device, dtype=dtype)
    cur_v = torch.empty(len(seq_lens), KVH, D, device=device, dtype=dtype)
    for si, sl in enumerate(seq_lens):
        last = sl - 1
        for lb in range(bps[si]):
            pb = phys[nxt]; nxt += 1
            block_table[si, lb] = pb
            for j in range(BS):
                tok = lb * BS + j
                if tok < sl and (tok < last or prefill_last):
                    key_cache[pb, :, j, :] = kv[si][0][tok]
                    value_cache[pb, :, j, :] = kv[si][1][tok]
        # decode token -> its slot
        lb_last = last // BS
        off_last = last % BS
        pb_last = int(block_table[si, lb_last])
        slot_map.append(pb_last * BS + off_last)
        cur_k[si] = kv[si][0][last]
        cur_v[si] = kv[si][1][last]
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    qsl = torch.arange(len(seq_lens) + 1, device=device, dtype=torch.int32)  # q_len==1 each
    query = torch.cat(queries, dim=0)
    slot_mapping = torch.tensor(slot_map, device=device, dtype=torch.int64)
    tsi = torch.arange(len(seq_lens), device=device, dtype=torch.int32)
    md = CustomTritonMetadata(
        num_actual_tokens=len(seq_lens), query_start_loc=qsl, seq_lens=seq_lens_t,
        block_table=block_table, slot_mapping=slot_mapping, token_seq_idx=tsi,
        causal=True, max_query_len=1, max_seq_len=max(seq_lens))
    return dict(kv=kv, query=query, key=cur_k, value=cur_v, packed=packed,
                key_cache=key_cache, value_cache=value_cache, block_table=block_table,
                md=md, seq_lens=seq_lens, num_tokens=len(seq_lens), qsl=qsl,
                seq_lens_t=seq_lens_t, tsi=tsi)


def reference_decode(kv, queries, scale):
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


def run_forward(impl, layer, d, dtype=torch.bfloat16):
    """Drive the REAL CustomTritonImpl.forward with the forward output contract."""
    out = torch.empty(d["num_tokens"], QH * D, device=d["query"].device, dtype=dtype)
    impl.forward(layer=layer, query=d["query"], key=d["key"], value=d["value"],
                 kv_cache=d["packed"], attn_metadata=d["md"], output=out)
    return out.view(d["num_tokens"], QH, D)


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    flag = os.environ.get("LIUXIAOCHEN_PAGED_DECODE_V3", "0")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    print(f"device_uuid: {torch.cuda.get_device_properties(dev).uuid}")
    print(f"flag LIUXIAOCHEN_PAGED_DECODE_V3={flag} v3_enabled={v4.v3_enabled()}")
    scale = 1.0 / (D ** 0.5)
    layer = _Layer(dev)
    impl = CustomTritonImpl(num_heads=QH, head_size=D, scale=scale, num_kv_heads=KVH)
    h0 = v4.STATS.hits

    # ---------- A. feature OFF -> real forward uses Triton read path ----------
    print("\n[A] feature OFF (real forward -> Triton read path)")
    os.environ["LIUXIAOCHEN_PAGED_DECODE_V3"] = "0"
    dA = build_decode([8192], dev, seed=10)
    refA = reference_decode(dA["kv"], [dA["query"][0:1]], scale)
    hits_before = v4.STATS.hits
    outA = run_forward(impl, layer, dA); torch.cuda.synchronize()
    maA = (outA.float() - refA).abs().max().item()
    check("[A] flag=0 forward vs reference <=5e-3 (Triton path)", maA <= STRICT, f"max_abs={maA:.3e}")
    check("[A] flag=0 V3 hit count unchanged", v4.STATS.hits == hits_before,
          f"hits {hits_before}->{v4.STATS.hits}")
    check("[A] flag=0 no NaN/Inf", not torch.isnan(outA).any() and not torch.isinf(outA).any())

    # ---------- B. feature ON -> real forward hits V3 ----------
    print("\n[B] feature ON (real forward -> V3 dispatch)")
    os.environ["LIUXIAOCHEN_PAGED_DECODE_V3"] = "1"
    for nm, sls, seed in [("ns1_sl128", [128], 20), ("ns1_sl8192", [8192], 21),
                          ("ns4_mixed", [128, 512, 4096, 8192], 22)]:
        d = build_decode(sls, dev, seed=seed)
        ref = reference_decode(d["kv"], [d["query"][i:i+1] for i in range(len(sls))], scale)
        hits_before = v4.STATS.hits
        out_fwd = run_forward(impl, layer, d); torch.cuda.synchronize()
        check(f"[B:{nm}] forward hit V3 (hit count +1)", v4.STATS.hits == hits_before + 1,
              f"hits {hits_before}->{v4.STATS.hits}")
        # direct V3 on the SAME (now fully-written) cache views
        out_direct = torch.empty(d["num_tokens"], QH, D, device=dev, dtype=torch.bfloat16)
        paged_decode_v3(query=d["query"], key_cache=d["key_cache"], value_cache=d["value_cache"],
                        output=out_direct, query_start_loc=d["qsl"], seq_lens=d["seq_lens_t"],
                        token_seq_idx=d["tsi"], block_table=d["block_table"], scale=scale,
                        split_size_tokens=256, max_seq_len=max(sls))
        torch.cuda.synchronize()
        check(f"[B:{nm}] forward(V3) == direct V3 bit-identical",
              torch.equal(out_fwd, out_direct),
              f"max_abs={(out_fwd.float()-out_direct.float()).abs().max().item():.3e}")
        ma = (out_fwd.float() - ref).abs().max().item()
        check(f"[B:{nm}] forward(V3) vs reference <=5e-3", ma <= STRICT, f"max_abs={ma:.3e}")
        check(f"[B:{nm}] output shape/dtype contract",
              tuple(out_fwd.shape) == (d["num_tokens"], QH, D) and out_fwd.dtype == torch.bfloat16)

    # ---------- C. Prefill under feature ON -> Triton fallback ----------
    print("\n[C] Prefill (feature ON -> Triton fallback, no V3)")
    os.environ["LIUXIAOCHEN_PAGED_DECODE_V3"] = "1"
    sl = 128
    torch.manual_seed(30)
    k_full = torch.randn(sl, KVH, D, device=dev, dtype=torch.bfloat16)
    v_full = torch.randn(sl, KVH, D, device=dev, dtype=torch.bfloat16)
    q_full = torch.randn(sl, QH, D, device=dev, dtype=torch.bfloat16)
    nb = _ceil(sl, BS) + 2
    packed = torch.randn(nb, KVH, BS, 2 * D, device=dev, dtype=torch.bfloat16)
    kc, vc = packed.split(D, dim=-1)
    bt = torch.zeros(1, _ceil(sl, BS), device=dev, dtype=torch.int32)
    slot_map = []
    for j in range(sl):
        lb, off = j // BS, j % BS
        pb = lb  # identity mapping for prefill
        bt[0, lb] = pb
        slot_map.append(pb * BS + off)
    qsl = torch.tensor([0, sl], device=dev, dtype=torch.int32)
    tsi = torch.zeros(sl, device=dev, dtype=torch.int32)
    md = CustomTritonMetadata(num_actual_tokens=sl, query_start_loc=qsl,
                              seq_lens=torch.tensor([sl], device=dev, dtype=torch.int32),
                              block_table=bt, slot_mapping=torch.tensor(slot_map, device=dev, dtype=torch.int64),
                              token_seq_idx=tsi, causal=True, max_query_len=sl, max_seq_len=sl)
    dP = dict(query=q_full, key=k_full, value=v_full, packed=packed, num_tokens=sl, md=md)
    # causal reference
    refP = torch.empty(sl, QH, D, dtype=torch.float32, device=dev)
    qf, kf, vf = q_full.float(), k_full.float(), v_full.float()
    for i in range(sl):
        for h in range(QH):
            kvh = h // (QH // KVH)
            s = (qf[i, h] * scale) @ kf[:i+1, kvh].T
            w = torch.softmax(s, dim=-1)
            refP[i, h] = w @ vf[:i+1, kvh]
    hits_before = v4.STATS.hits
    fb_before = v4.STATS.fallbacks
    out_p = run_forward(impl, layer, dP); torch.cuda.synchronize()
    maP = (out_p.float() - refP).abs().max().item()
    check("[C] prefill: V3 hit count unchanged (fallback)", v4.STATS.hits == hits_before,
          f"hits {hits_before}->{v4.STATS.hits}")
    check("[C] prefill: fallback recorded", v4.STATS.fallbacks > fb_before,
          f"fallbacks {fb_before}->{v4.STATS.fallbacks}")
    check("[C] prefill forward vs causal reference (tutorial 2e-2, Triton fallback path)",
          maP <= TRITON_TOL, f"max_abs={maP:.3e} (bf16 Triton path; strict-5e-3 too tight here)")
    check("[C] prefill no NaN/Inf", not torch.isnan(out_p).any() and not torch.isinf(out_p).any())

    # ---------- D. Unsupported (fp16 decode) under feature ON -> fallback ----------
    print("\n[D] Unsupported fp16 decode (feature ON -> Triton fallback)")
    os.environ["LIUXIAOCHEN_PAGED_DECODE_V3"] = "1"
    dD = build_decode([256], dev, seed=40, dtype=torch.float16)
    refD = reference_decode(dD["kv"], [dD["query"][0:1]], scale)
    hits_before = v4.STATS.hits
    out_d = run_forward(impl, layer, dD, dtype=torch.float16); torch.cuda.synchronize()
    maD = (out_d.float() - refD).abs().max().item()
    check("[D] fp16: V3 hit count unchanged (fallback)", v4.STATS.hits == hits_before,
          f"hits {hits_before}->{v4.STATS.hits}")
    check("[D] fp16 forward vs reference (tutorial 2e-2, Triton fallback path)",
          maD <= TRITON_TOL, f"max_abs={maD:.3e}")
    check("[D] fp16 no NaN/Inf", not torch.isnan(out_d).any() and not torch.isinf(out_d).any())

    # ---------- E. Repeated A->B->A through real forward ----------
    print("\n[E] repeated A->B->A (real forward, feature ON)")
    os.environ["LIUXIAOCHEN_PAGED_DECODE_V3"] = "1"
    dEA = build_decode([8192], dev, seed=50)
    dEB = build_decode([8192], dev, seed=51)
    oEA1 = run_forward(impl, layer, dEA)
    oEB = run_forward(impl, layer, dEB)
    oEA2 = run_forward(impl, layer, dEA)
    torch.cuda.synchronize()
    check("[E] A1==A2 via forward (no first-tensor capture)", torch.equal(oEA1, oEA2))
    check("[E] A!=B via forward", not torch.equal(oEA1, oEB),
          f"max_abs={(oEA1.float()-oEB.float()).abs().max().item():.3e}")

    total_hits = v4.STATS.hits - h0
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\nV3 hits this run={total_hits} fallbacks={v4.STATS.fallbacks} "
          f"reasons={v4.STATS.reason_counts}")
    print(f"SUMMARY: {len(_results)-n_fail}/{len(_results)} PASS, {n_fail} FAIL")
    print("BACKEND_FORWARD_GATE=" + ("PASS" if n_fail == 0 else "FAIL"))
    sys.exit(0 if n_fail == 0 else 2)


if __name__ == "__main__":
    main()
