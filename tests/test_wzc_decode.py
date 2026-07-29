# SPDX-License-Identifier: Apache-2.0
"""
Correctness test for the wzc PAGED DECODE kernel on the vLLM paged interface.
=============================================================================

Reuses the tutorial's paged-KV-cache builder + naive PyTorch reference and drives
``paged_attention_wzc`` with DECODE requests (query_len == 1, seq_len > 1). The
adapter routes these to the wzc paged decode kernel (FlashDecoding split-KV,
page_size=128), gathering each request's history out of the vLLM paged cache
(block_size=16 here) and repacking it into the kernel's 128-page pool.

Covers variable seq_len (including non-128-aligned tails) and a mixed
prefill+decode batch. Target GQA family: q_heads=64, kv_heads=8, head_dim=128.

Run:
  cd /dockerdata/wangzicheng/vllm-custom-attention-tutorial
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_wzc_decode.py
"""

import torch

from tests.test_paged_attn_correctness import make_paged_kv_cache, naive_reference
from custom_backend.wzc_sparse_attention import paged_attention_wzc


def run_case(name, seqs, num_heads, num_kv_heads, head_size, block_size,
             device, dtype, rtol=2e-2, atol=2e-2):
    """seqs: list[(seq_len, query_len)]."""
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
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    print(f"[{name}] max_abs={max_err:.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"
    dtype = torch.bfloat16
    H, HK, hd, bs = 64, 8, 128, 16
    all_ok = True

    # Pure decode (q_len==1) at various seq_len -> wzc paged decode kernel.
    # Includes 128-aligned (256) and non-aligned (200, 1000, 4097) tails.
    all_ok &= run_case("decode-256",  [(256, 1)],  H, HK, hd, bs, device, dtype)
    all_ok &= run_case("decode-200",  [(200, 1)],  H, HK, hd, bs, device, dtype)
    all_ok &= run_case("decode-1000", [(1000, 1)], H, HK, hd, bs, device, dtype)
    all_ok &= run_case("decode-4097", [(4097, 1)], H, HK, hd, bs, device, dtype)
    # Batch of decode requests with different seq_len.
    all_ok &= run_case("decode-batch", [(300, 1), (777, 1), (128, 1)],
                       H, HK, hd, bs, device, dtype)
    # Mixed: a prefill request (sparse kernel) + a decode request (paged kernel).
    all_ok &= run_case("mixed-prefill-decode", [(256, 256), (500, 1)],
                       H, HK, hd, bs, device, dtype)

    print("=" * 50)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
