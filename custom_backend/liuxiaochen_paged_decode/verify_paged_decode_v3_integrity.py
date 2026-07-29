#!/usr/bin/env python3
"""V3.1 measurement-integrity audit for the cp.async paged-KV decode (Liu Xiaochen).

This is a READ-ONLY correctness/integrity harness. It does NOT change any V3/V2
kernel math, does NOT add PDL or pipeline stages, and does NOT touch the CUSTOM
backend. It answers one question: are the V3 measurements trustworthy, i.e. does
`paged_decode_v3` truly operate on the tensors passed to each call (no captured
pointers, no stale workspace), and does it stay bit-identical to V2 and within
5e-3 of the PyTorch reference?

Config (fixed): seq_len=8192, num_seqs=1, split=256, shuffled block_table, bf16.

Checks:
  1. pointer rebinding      -- A1==A2 bit-identical, A!=B, same compiled kernel reused
  2. A -> B -> A            -- interleave two independent tensor sets, no cross-talk
  3. K/V data replacement   -- same tensor object, swapped contents => output follows contents
  4. workspace NaN poisoning-- fill partial/lse with NaN before call; result must be unchanged
  5. output pointer         -- output.data_ptr() is the caller's buffer, unchanged across calls
  6. V3 vs V2               -- torch.equal (bit-identical)
  7. V3 vs PyTorch reference-- max_abs <= 5e-3

Empty-split neutrality is exercised with max_seq_len > seq_len (num_splits_max >
valid_splits): empty splits must write partial_o=0 and lse=-inf, and the poisoned
workspace must not leak into the result.

Run:
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python -u \
    custom_backend/liuxiaochen_paged_decode/verify_paged_decode_v3_integrity.py
"""

import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

from custom_backend.liuxiaochen_paged_decode import paged_decode_v2, paged_decode_v3
from custom_backend.liuxiaochen_paged_decode import runner_v3
from custom_backend.liuxiaochen_paged_decode.verify_paged_decode_v3 import (
    build_paged, naive_reference, build_metadata, QH, D,
)

SPLIT = 256
SEQ_LEN = 8192
STRICT = 5e-3

_results = []


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _ceil_div(a, b):
    return (a + b - 1) // b


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    print(f"device_uuid: {torch.cuda.get_device_properties(dev).uuid}")
    print(f"cfg: seq_len={SEQ_LEN} num_seqs=1 split={SPLIT} shuffle=True strict_ref={STRICT}")
    scale = 1.0 / (D ** 0.5)
    seq_lens = [SEQ_LEN]
    max_seq_len = SEQ_LEN
    nsm = _ceil_div(max_seq_len, SPLIT)

    # ---- two fully independent tensor sets (same shape/stride, different data) ----
    kvA, qA, kcA, vcA, btA = build_paged(seq_lens, dev, torch.bfloat16, shuffle=True, seed=0)
    kvB, qB, kcB, vcB, btB = build_paged(seq_lens, dev, torch.bfloat16, shuffle=True, seed=12345)
    queryA, qslA, slA, tsiA, ntA = build_metadata(seq_lens, qA, dev)
    queryB, qslB, slB, tsiB, ntB = build_metadata(seq_lens, qB, dev)

    same_shape = (
        queryA.shape == queryB.shape and kcA.shape == kcB.shape
        and vcA.shape == vcB.shape and btA.shape == btB.shape
        and queryA.stride() == queryB.stride() and kcA.stride() == kcB.stride()
        and vcA.stride() == vcB.stride() and btA.stride() == btB.stride()
    )
    indep_alloc = (
        queryA.data_ptr() != queryB.data_ptr()
        and kcA.data_ptr() != kcB.data_ptr()
        and vcA.data_ptr() != vcB.data_ptr()
    )

    outA = torch.empty(ntA, QH, D, device=dev, dtype=torch.bfloat16)
    outB = torch.empty(ntB, QH, D, device=dev, dtype=torch.bfloat16)
    op_A0 = outA.data_ptr()
    op_B0 = outB.data_ptr()
    kcA0, vcA0 = kcA.data_ptr(), vcA.data_ptr()

    refA = torch.cat(naive_reference(kvA, qA, scale), dim=0)
    refB = torch.cat(naive_reference(kvB, qB, scale), dim=0)

    # workspace buffer for this shape (allocates + returns the SAME object v3 uses)
    part_ws, lse_ws = runner_v3._workspace(1, QH, nsm, D, dev)

    print("data_ptrs (set A):")
    print(f"  query       = {hex(queryA.data_ptr())}")
    print(f"  key_cache   = {hex(kcA0)}")
    print(f"  value_cache = {hex(vcA0)}")
    print(f"  output      = {hex(op_A0)}")
    print(f"  block_table = {hex(btA.data_ptr())}")
    print(f"  workspace   = {hex(part_ws.data_ptr())} (partial) / {hex(lse_ws.data_ptr())} (lse)")
    print("data_ptrs (set B):")
    print(f"  query       = {hex(queryB.data_ptr())}")
    print(f"  key_cache   = {hex(kcB.data_ptr())}")
    print(f"  value_cache = {hex(vcB.data_ptr())}")
    print(f"  output      = {hex(op_B0)}")
    print(f"  block_table = {hex(btB.data_ptr())}")

    # ---- V2 baselines (original A/B data) ----
    outA_v2 = torch.empty_like(outA)
    outB_v2 = torch.empty_like(outB)
    paged_decode_v2(query=queryA, key_cache=kcA, value_cache=vcA, output=outA_v2,
                    query_start_loc=qslA, seq_lens=slA, token_seq_idx=tsiA, block_table=btA,
                    scale=scale, split_size_tokens=SPLIT, max_seq_len=max_seq_len)
    paged_decode_v2(query=queryB, key_cache=kcB, value_cache=vcB, output=outB_v2,
                    query_start_loc=qslB, seq_lens=slB, token_seq_idx=tsiB, block_table=btB,
                    scale=scale, split_size_tokens=SPLIT, max_seq_len=max_seq_len)

    def v3A(out=None, msl=max_seq_len):
        return paged_decode_v3(query=queryA, key_cache=kcA, value_cache=vcA,
                               output=outA if out is None else out,
                               query_start_loc=qslA, seq_lens=slA, token_seq_idx=tsiA,
                               block_table=btA, scale=scale, split_size_tokens=SPLIT,
                               max_seq_len=msl)

    def v3B():
        return paged_decode_v3(query=queryB, key_cache=kcB, value_cache=vcB, output=outB,
                               query_start_loc=qslB, seq_lens=slB, token_seq_idx=tsiB,
                               block_table=btB, scale=scale, split_size_tokens=SPLIT,
                               max_seq_len=max_seq_len)

    n_compile_start = len(runner_v3._compile_cache)

    # ============ 1+2. pointer rebinding via A -> B -> A ============
    print("\n[1+2] pointer rebinding (A -> B -> A)")
    v3A(); a1 = outA.clone()
    n_after_a1 = len(runner_v3._compile_cache)
    v3B(); b = outB.clone()
    n_after_b = len(runner_v3._compile_cache)
    v3A(); a2 = outA.clone()
    n_after_a2 = len(runner_v3._compile_cache)
    torch.cuda.synchronize()

    check("A1 == A2 bit-identical", torch.equal(a1, a2))
    check("A output != B output (no cross-talk)", not torch.equal(a1, b),
          f"max_abs(A,B)={_max_abs(a1, b):.3e}")
    check("compiled kernel cache reused for B (same shape)", n_after_b == n_after_a1,
          f"start={n_compile_start} afterA1={n_after_a1} afterB={n_after_b}")
    check("compiled kernel cache reused for A2", n_after_a2 == n_after_a1,
          f"afterA2={n_after_a2}")
    check("no first-call tensor-pointer capture (A1==A2 and A!=B)",
          torch.equal(a1, a2) and not torch.equal(a1, b))

    # ============ 5. output pointer identity ============
    print("\n[5] output pointer identity")
    check("output A ptr unchanged across A1/A2", outA.data_ptr() == op_A0, f"{hex(outA.data_ptr())}")
    check("output B ptr unchanged", outB.data_ptr() == op_B0, f"{hex(outB.data_ptr())}")

    # ============ 6. V3 vs V2 bit-identical ============
    print("\n[6] V3 vs V2 bit-identical")
    check("A: V3 == V2 (torch.equal)", torch.equal(a1, outA_v2), f"max_abs={_max_abs(a1, outA_v2):.3e}")
    check("B: V3 == V2 (torch.equal)", torch.equal(b, outB_v2), f"max_abs={_max_abs(b, outB_v2):.3e}")

    # ============ 7. V3 vs PyTorch reference ============
    print("\n[7] V3 vs PyTorch reference (max_abs <= 5e-3)")
    maA, maB = _max_abs(a1, refA), _max_abs(b, refB)
    check("A: max_abs(V3, ref) <= 5e-3", maA <= STRICT, f"max_abs={maA:.3e}")
    check("B: max_abs(V3, ref) <= 5e-3", maB <= STRICT, f"max_abs={maB:.3e}")

    # ============ 3. K/V data replacement (same object, swapped contents) ============
    print("\n[3] K/V data replacement (in-place, pointer stable)")
    kA_save, vA_save = kcA.clone(), vcA.clone()
    kcA.copy_(torch.randn_like(kcA))
    vcA.copy_(torch.randn_like(vcA))
    v3A(); a_mod = outA.clone(); torch.cuda.synchronize()
    check("K/V contents replaced -> output changes", not torch.equal(a_mod, a1),
          f"max_abs(mod,A1)={_max_abs(a_mod, a1):.3e}")
    check("K/V data_ptr stable after in-place replace",
          kcA.data_ptr() == kcA0 and vcA.data_ptr() == vcA0)
    kcA.copy_(kA_save)
    vcA.copy_(vA_save)
    v3A(); a_res = outA.clone(); torch.cuda.synchronize()
    check("K/V restored -> output == A1 (live read + determinism)", torch.equal(a_res, a1))

    # ============ 4a. workspace NaN poisoning (main config, no empty splits) ============
    print("\n[4a] workspace NaN poisoning (main config)")
    part_ws.fill_(float("nan"))
    lse_ws.fill_(float("nan"))
    v3A(); a_poison = outA.clone(); torch.cuda.synchronize()
    check("poisoned workspace -> output == A1 (result independent of prior workspace)",
          torch.equal(a_poison, a1))
    check("poisoned-run output has no NaN", not torch.isnan(a_poison).any())
    check("poisoned-run output has no Inf", not torch.isinf(a_poison).any())

    # ============ 4b. workspace poisoning + empty-split neutrality ============
    print("\n[4b] workspace NaN poisoning + empty-split neutrality (max_seq_len > seq_len)")
    max_big = SEQ_LEN + 2048           # 10240
    nsm_big = _ceil_div(max_big, SPLIT)      # 40
    valid_splits = _ceil_div(SEQ_LEN, SPLIT)  # 32
    part_big, lse_big = runner_v3._workspace(1, QH, nsm_big, D, dev)
    part_big.fill_(float("nan"))
    lse_big.fill_(float("nan"))
    out_big = torch.empty_like(outA)
    v3A(out=out_big, msl=max_big)
    torch.cuda.synchronize()
    valid_part = part_big[0, :, :valid_splits, :]
    valid_lse = lse_big[0, :, :valid_splits]
    empty_part = part_big[0, :, valid_splits:nsm_big, :]
    empty_lse = lse_big[0, :, valid_splits:nsm_big]
    nz_empty = int(torch.count_nonzero(empty_part).item())
    check("valid splits fully covered (no NaN in partial_o)", not torch.isnan(valid_part).any())
    check("valid splits lse have no NaN", not torch.isnan(valid_lse).any())
    check("empty split partial_o == 0", nz_empty == 0, f"nonzero_entries={nz_empty}")
    check("empty split lse == -inf", bool((empty_lse == float("-inf")).all()))
    check("empty-split run output no NaN/Inf",
          (not torch.isnan(out_big).any()) and (not torch.isinf(out_big).any()))
    check("empty-split run output == A1 (empty splits contribute nothing)",
          torch.equal(out_big, a1))
    ma_big = _max_abs(out_big, refA)
    check("empty-split run max_abs(V3, ref) <= 5e-3", ma_big <= STRICT, f"max_abs={ma_big:.3e}")

    # ============ tensor-independence bookkeeping ============
    print("\n[bookkeeping] tensor independence")
    check("A/B share shape & stride", same_shape)
    check("A/B independently allocated (distinct data_ptr)", indep_alloc)

    # ---- summary ----
    n_fail = sum(1 for _, ok in _results if not ok)
    n_pass = len(_results) - n_fail
    print(f"\nSUMMARY: {n_pass}/{len(_results)} PASS, {n_fail} FAIL")
    print("INTEGRITY_AUDIT=" + ("PASS" if n_fail == 0 else "FAIL"))
    sys.exit(0 if n_fail == 0 else 2)


if __name__ == "__main__":
    main()
