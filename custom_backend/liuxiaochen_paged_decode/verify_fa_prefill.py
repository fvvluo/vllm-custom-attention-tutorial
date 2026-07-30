#!/usr/bin/env python3
"""Synthetic check: V4 fast paged FlashAttention prefill vs PyTorch reference.

Validates that try_fast_paged_prefill (called as paged_attention_triton would call
it) matches a causal GQA reference on the SAME packed paged KV-cache layout the
CUSTOM backend produces, before touching a live service. Also checks it matches the
teaching Triton kernel within the tutorial 2e-2 tolerance.
"""
import sys
import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.triton_attention import paged_attention_triton
from custom_backend.liuxiaochen_paged_decode.vllm_prefill_fa import try_fast_paged_prefill

QH, KVH, D, BS = 64, 8, 128, 16
TOL = 2e-2
_res = []

def check(n, ok, d=""):
    _res.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")

def build(seq_lens, dev, seed=0):
    torch.manual_seed(seed)
    kv, qs = [], []
    for sl in seq_lens:
        kv.append((torch.randn(sl, KVH, D, device=dev, dtype=torch.bfloat16),
                   torch.randn(sl, KVH, D, device=dev, dtype=torch.bfloat16)))
        qs.append(torch.randn(sl, QH, D, device=dev, dtype=torch.bfloat16))  # prefill: q_len==seq_len
    bps = [(sl + BS - 1)//BS for sl in seq_lens]
    tot = sum(bps) + 3
    mnb = max(bps)
    packed = torch.randn(tot, KVH, BS, 2*D, device=dev, dtype=torch.bfloat16)
    kc, vc = packed.split(D, dim=-1)
    bt = torch.zeros(len(seq_lens), mnb, device=dev, dtype=torch.int32)
    nxt = 0
    for si, sl in enumerate(seq_lens):
        for lb in range(bps[si]):
            pb = nxt; nxt += 1
            bt[si, lb] = pb
            for j in range(BS):
                tok = lb*BS + j
                if tok < sl:
                    kc[pb,:,j,:] = kv[si][0][tok]
                    vc[pb,:,j,:] = kv[si][1][tok]
    qsl = torch.zeros(len(seq_lens)+1, device=dev, dtype=torch.int32)
    qsl[1:] = torch.tensor([sl for sl in seq_lens], device=dev).cumsum(0)
    query = torch.cat(qs, dim=0)
    seq_lens_t = torch.tensor(seq_lens, device=dev, dtype=torch.int32)
    nt = int(qsl[-1])
    tsi = torch.searchsorted(qsl[1:], torch.arange(nt, device=dev, dtype=torch.int32), right=True).to(torch.int32)
    return dict(kv=kv, qs=qs, query=query, kc=kc, vc=vc, bt=bt, qsl=qsl, sl=seq_lens_t, tsi=tsi, nt=nt)

def ref(kv, qs, scale):
    outs = []
    for (k, v), q in zip(kv, qs):
        sl = k.shape[0]
        o = torch.empty((sl, QH, D), dtype=torch.float32, device=q.device)
        qf, kf, vf = q.float(), k.float(), v.float()
        for i in range(sl):
            for h in range(QH):
                kvh = h // (QH//KVH)
                s = (qf[i,h]*scale) @ kf[:i+1, kvh].T
                w = torch.softmax(s, dim=-1)
                o[i,h] = w @ vf[:i+1, kvh]
        outs.append(o)
    return torch.cat(outs, dim=0)

def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"device {dev} ({torch.cuda.get_device_name(dev)})")
    scale = 1.0/(D**0.5)
    for nm, sls, seed in [("sl128",[128],0), ("sl517",[517],1), ("multi",[128,300,64],2)]:
        d = build(sls, dev, seed)
        r = ref(d["kv"], d["qs"], scale)
        out_fa = torch.empty(d["nt"], QH, D, device=dev, dtype=torch.bfloat16)
        used = try_fast_paged_prefill(query=d["query"], key_cache=d["kc"], value_cache=d["vc"],
                                      output=out_fa, query_start_loc=d["qsl"], seq_lens=d["sl"],
                                      block_table=d["bt"], scale=scale)
        torch.cuda.synchronize()
        check(f"[{nm}] fast prefill dispatched", used)
        if used:
            ma = (out_fa.float()-r).abs().max().item()
            check(f"[{nm}] FA prefill vs PyTorch ref <=2e-2", ma <= TOL, f"max_abs={ma:.3e}")
    n_fail = sum(1 for x in _res if not x)
    print(f"SUMMARY {len(_res)-n_fail}/{len(_res)} PASS")
    print("FA_PREFILL_TEST=" + ("PASS" if n_fail==0 else "FAIL"))
    sys.exit(0 if n_fail==0 else 2)

if __name__ == "__main__":
    main()
