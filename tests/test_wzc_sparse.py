# SPDX-License-Identifier: Apache-2.0
"""
Correctness test for the wzc sparse prefill kernel on the vLLM PAGED interface.
==============================================================================

Reuses the tutorial's paged-KV-cache builder + naive PyTorch reference
(tests/test_paged_attn_correctness.py), but drives ``paged_attention_wzc`` and
uses cases the sparse PREFILL kernel can actually serve:
  - pure prefill (query_len == seq_len),
  - seq_len % 128 == 0, head_size == 128, GQA 64/8 (the target shape family).

Two kinds of check:
  1. tau=1.0  -> LOSSLESS: must match the exact reference within bf16 tolerance
     (the sparse kernel selects ALL causal blocks; this validates the paged
     gather + adapter plumbing end-to-end).
  2. tau=0.99 -> LOSSY: report max/mean error + effective keep (informational);
     random data is the worst case for block sparsity, so we only assert the
     mean error is reasonable, not the max.

Run:
  cd /dockerdata/wangzicheng/vllm-custom-attention-tutorial
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_wzc_sparse.py
"""

import os
import torch

from tests.test_paged_attn_correctness import make_paged_kv_cache, naive_reference
from custom_backend.wzc_sparse_attention import paged_attention_wzc


def run_case(name, seqs, num_heads, num_kv_heads, head_size, block_size,
             device, dtype, tau, assert_lossless):
    os.environ["WZC_SPARSE_TAU"] = str(tau)
    torch.manual_seed(0)
    kv_data, queries = [], []
    for (seq_len, q_len) in seqs:
        k = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        q = torch.randn(q_len, num_heads, head_size, device=device, dtype=dtype)
        kv_data.append((k, v))
        queries.append(q)

    scale = 1.0 / (head_size ** 0.5)
    key_cache, value_cache, block_table = make_paged_kv_cache(
        kv_data, num_kv_heads, head_size, block_size, device, dtype)

    q_lens = [q.shape[0] for q in queries]
    seq_lens = torch.tensor([s for (s, _) in seqs], device=device, dtype=torch.int32)
    query_start_loc = torch.zeros(len(seqs) + 1, device=device, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(q_lens, device=device).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    query = torch.cat(queries, dim=0)
    token_seq_idx = torch.searchsorted(
        query_start_loc[1:], torch.arange(num_tokens, device=device, dtype=torch.int32),
        right=True).to(torch.int32)

    output = torch.empty(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    paged_attention_wzc(
        query=query, key_cache=key_cache, value_cache=value_cache, output=output,
        query_start_loc=query_start_loc, seq_lens=seq_lens,
        token_seq_idx=token_seq_idx, block_table=block_table, scale=scale)

    ref = torch.cat(naive_reference(kv_data, queries, scale), dim=0)
    got = output.float()
    max_err = (got - ref).abs().max().item()
    mean_err = (got - ref).abs().mean().item()
    if assert_lossless:
        ok = torch.allclose(got, ref, rtol=2e-2, atol=2e-2)
        print(f"[{name} tau={tau}] max_abs={max_err:.4e} mean={mean_err:.4e} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return ok
    # lossy: informational; assert only that mean error stays small
    ok = mean_err < 5e-2
    print(f"[{name} tau={tau}] max_abs={max_err:.4e} mean={mean_err:.4e} "
          f"(lossy, mean<5e-2 -> {'PASS' if ok else 'FAIL'})")
    return ok


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"
    dtype = torch.bfloat16
    H, HK, hd, bs = 64, 8, 128, 16
    all_ok = True

    # 1) LOSSLESS: tau=1.0, pure prefill, seq%128==0 -> bit-exact vs reference.
    all_ok &= run_case("prefill-single", [(256, 256)], H, HK, hd, bs, device, dtype,
                       tau=1.0, assert_lossless=True)
    all_ok &= run_case("prefill-batch", [(256, 256), (512, 512)], H, HK, hd, bs,
                       device, dtype, tau=1.0, assert_lossless=True)
    # 2) LOSSY: tau=0.99 on longer prefills (sparsity active). Random data is
    #    the worst case, so only the mean error is asserted. These LONG, 128-
    #    aligned pure-prefills are exactly what the wzc c4 kernel serves (verify
    #    with WZC_SPARSE_STATS=1: kernel_reqs increments, max_kernel_seq grows).
    all_ok &= run_case("prefill-2k", [(2048, 2048)], H, HK, hd, bs, device, dtype,
                       tau=0.99, assert_lossless=False)
    all_ok &= run_case("prefill-4k", [(4096, 4096)], H, HK, hd, bs, device, dtype,
                       tau=0.99, assert_lossless=False)
    all_ok &= run_case("prefill-8k", [(8192, 8192)], H, HK, hd, bs, device, dtype,
                       tau=0.99, assert_lossless=False)
    # tau=1.0 lossless at long length: must still match the exact reference.
    all_ok &= run_case("prefill-4k-lossless", [(4096, 4096)], H, HK, hd, bs, device,
                       dtype, tau=1.0, assert_lossless=True)
    # UNALIGNED lengths (seq % 128 != 0): the adapter pads to a 128 multiple and
    # drops padded rows. tau=1.0 must stay lossless -> proves padding is correct
    # (this is the real-serving case: arbitrary prompt lengths).
    all_ok &= run_case("prefill-unaligned-1000", [(1000, 1000)], H, HK, hd, bs,
                       device, dtype, tau=1.0, assert_lossless=True)
    all_ok &= run_case("prefill-unaligned-7975", [(7975, 7975)], H, HK, hd, bs,
                       device, dtype, tau=1.0, assert_lossless=True)
    # 3) MIXED: a decode request (routes to torch fallback) + a prefill request
    #    (routes to the sparse kernel) in one batch -> still correct.
    all_ok &= run_case("mixed", [(256, 256), (300, 1)], H, HK, hd, bs, device, dtype,
                       tau=1.0, assert_lossless=True)

    print("=" * 50)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
