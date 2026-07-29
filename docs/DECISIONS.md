# DECISIONS

## D1. Use /usr/bin/python, not conda
Tutorial requires the system Python whose wheel provides the exact vLLM
0.18.1rc1.dev3933+ga49d37c6b build. conda python would not have the matching
compiled extensions. All commands use `/usr/bin/python`.

## D2. Personalized scripts instead of editing originals
Copied `setup_vllm_source.sh`, `serve_qwen3_flashattn.sh`, `serve_qwen3_custom.sh`
to `*_liuxiaochen.sh` with only path parameterization + safety guards. Original
scripts (author landojiang) left byte-for-byte unchanged, so the repo stays
diffable and other members' references remain valid.

## D3. Symlink .so, don't recompile
The wheel already ships the exact-commit CUDA extensions. Setup symlinks the 16
`.so` (+ `_version.py`) into the source tree; `PYTHONPATH=$VLLM_SRC` overrides
only the `.py`. This lets us edit vLLM Python (future custom backend) without a
multi-hour CUDA rebuild.

## D4. Do NOT create a root-level auto-registered ops module
The tutorial's plugin mechanism (`vllm.general_plugins` entry point) already
handles CUSTOM backend registration via `pip install -e .`. This is the tutorial-
native path and does not pollute anything.

## D5. Baseline correctness tolerance kept at tutorial value
bf16 rtol=atol=2e-2, unchanged. All paged-attn scenarios pass comfortably
(max 7.81e-3 << 2e-2).

---

## Interface audit: `paged_attention_triton(...)` (Part 3 kernel contract)

Signature (BHSD-free; vLLM packed-token + paged layout):

    paged_attention_triton(query, key_cache, value_cache, output,
                           query_start_loc, seq_lens, token_seq_idx,
                           block_table, scale)

| param | shape | dtype | meaning |
|---|---|---|---|
| query | [num_tokens, num_heads, head_size] | bf16 | ALL requests' query tokens packed/concatenated along dim0 |
| key_cache | [num_blocks, num_kv_heads, block_size, head_size] | bf16 | physical paged K pool |
| value_cache | same as key_cache | bf16 | physical paged V pool |
| output | [num_tokens, num_heads, head_size] | bf16 | written in-place, returned |
| query_start_loc | [num_seqs+1] | int32 | prefix-sum of per-request query_len; request r's tokens are [qsl[r], qsl[r+1]) |
| seq_lens | [num_seqs] | int32 | each request's TOTAL length (context + this query) |
| token_seq_idx | [num_tokens] | int32 | precomputed token→request-index map (avoids in-kernel search) |
| block_table | [num_seqs, max_num_blocks] | int32 | logical block → physical block number |
| scale | float | — | 1/sqrt(head_size) |

- device: all CUDA, same GPU. Strides passed explicitly to the kernel (query
  q_stride_t/q_stride_h, out o_stride_t/o_stride_h, cache kc/vc_stride_b/h/s,
  block_table_stride) — the kernel does NOT assume contiguity.
- grid = (num_tokens, num_heads): one Triton program per (token, head).
- prefill vs decode: NOT separate code paths. A request with query_len==seq_len
  is prefill; query_len==1 with context is decode; a batch may mix both. The
  kernel derives `context_len = seq_len - query_len` and
  `abs_pos = context_len + idx_in_query` for causal bound per token.
- causal position: each query token attends kv_pos in [0, abs_pos]; positions >
  abs_pos masked to -inf (online softmax).
- GQA: `kv_head_idx = head_idx // (num_heads // num_kv_heads)` (64→8, group 8).
- paged mapping: for kv_pos, `logical_block=kv_pos//block_size`,
  `slot=kv_pos%block_size`, `pb=block_table[seq_idx, logical_block]`, read
  `k_cache[pb, kv_head_idx, slot, :]`.
- KV cache WRITE happens in vLLM's model runner (not here); this kernel only
  READS the paged cache. output written in-place.

## Old B7 (attention-test) vs new vLLM interface

Reusable IDEAS (not code):
- warp-MMA m16n8k16 QK/PV Tensor-Core formulation
- online softmax (m/l/p, LSE)
- GQA head grouping (kv_head = q_head // group)
- Split-KV parallelism + LSE combine
- cp.async double-buffered K/V staging

NOT directly reusable:
- contiguous BHSD address math — new layout is a paged block pool with strides
- single-request, q_len==1 fixed interface — new is packed multi-request, mixed
  prefill/decode
- linear K/V reads — new must go through block_table (logical→physical)
- old cache-key / grid mapping — new grid is (num_tokens, num_heads); metadata
  (query_start_loc/seq_lens/token_seq_idx/block_table) drives everything

Conclusion: the numerical core (MMA attention + online softmax + LSE combine)
transfers; the memory addressing / batching / scheduling layer must be rewritten
for the paged, packed-token interface. No port done this phase (analysis only).
