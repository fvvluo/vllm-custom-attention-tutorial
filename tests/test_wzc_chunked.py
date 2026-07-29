# SPDX-License-Identifier: Apache-2.0
"""Correctness test for CHUNKED-prefill routing in the wzc adapter.

Drives ``paged_attention_wzc`` with requests that have BOTH context and multiple
query tokens (q_len>1, seq_len>q_len) — i.e. vLLM chunked-prefill chunks — which
now route to the RECTANGULAR-causal sparse kernel (ops/_wzc_attn_sparse_rect.py)
instead of the torch fallback. Compares against the tutorial's naive reference.

Covers:
  * 128-aligned context (kernel path): context in {128,2048,4096}, various q_len
    incl. non-128-aligned q_len (adapter pads q up to 128).
  * tau=1.0 must be lossless (== exact reference within bf16 tol).
  * a mixed batch (pure-prefill chunk0 + context chunk + decode) stays correct.

Run:
  cd /dockerdata/wangzicheng/vllm-custom-attention-tutorial
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_wzc_chunked.py
"""
import os
import torch

from tests.test_paged_attn_correctness import make_paged_kv_cache, naive_reference
from custom_backend.wzc_sparse_attention import paged_attention_wzc


def run_case(name, seqs, num_heads, num_kv_heads, head_size, block_size,
             device, dtype, tau=1.0, assert_lossless=True):
    """seqs: list[(seq_len, query_len)]. query at chunk covers abs positions
    [seq_len-query_len, seq_len)."""
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
    print(f"[{name} tau={tau}] max_abs={max_err:.4e} mean={mean_err:.4e} (lossy)")
    return True


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    device, dtype = "cuda", torch.bfloat16
    H, HK, hd, bs = 64, 8, 128, 16
    ok = True

    # CHUNKED prefill: context (128-aligned) + multi-query, tau=1 lossless.
    # (seq_len, q_len): context = seq_len - q_len.
    ok &= run_case("chunk-ctx128-q128",  [(256, 128)],   H, HK, hd, bs, device, dtype)
    ok &= run_case("chunk-ctx2048-q512", [(2560, 512)],  H, HK, hd, bs, device, dtype)
    ok &= run_case("chunk-ctx4096-q256", [(4352, 256)],  H, HK, hd, bs, device, dtype)
    # non-128-aligned q_len (adapter pads q up to 128); context still 128-aligned.
    ok &= run_case("chunk-ctx2048-q500", [(2548, 500)],  H, HK, hd, bs, device, dtype)
    ok &= run_case("chunk-ctx4096-q100", [(4196, 100)],  H, HK, hd, bs, device, dtype)
    # mixed batch: pure-prefill chunk0 + context chunk + decode.
    ok &= run_case("mixed-prefill-chunk-decode",
                   [(256, 256), (2560, 512), (300, 1)], H, HK, hd, bs, device, dtype)
    # lossy (informational; random data worst case)
    run_case("chunk-lossy", [(4352, 512)], H, HK, hd, bs, device, dtype,
             tau=0.99, assert_lossless=False)

    print("=" * 50)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
