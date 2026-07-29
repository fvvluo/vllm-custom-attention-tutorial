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

## D6. V2 timing convention (Phase V2.1)
- unique-KV GB/s uses TOTAL K+V bytes: 2*sum(seq_len)*Hkv*D*2(bf16). 128K K+V=512 MiB.
- Continuous B5/B7 under this convention: B5≈1.81 TB/s, B7≈2.88 TB/s (128K).
- End-to-end V2 timing includes the per-call lse.fill_(-inf) reset (honest);
  Stage-1-only / combine-only are measured via build_v2_runners with pre-reset
  workspace to isolate kernel cost. Stage-1+combine != end-to-end exactly (each
  has independent launch/event overhead).
- Best 128K split = 256 (num_splits=512, 4 tiles/CTA). combine ~0.3%, Stage-1 ~99%.
- block_table lookup is NOT claimed negligible — only "not a dominant bottleneck",
  pending cleaner profiling (shared-GPU contention contaminated 32768 identity run).
- V3 (cp.async) is justified: V2 sync load ≈167 GB/s << HBM roofline; overlap
  should close the gap toward B6/B7-style utilization.

## D7. V3 cp.async paged pipeline design (Phase V3)
- **ONE variable only**: V3 = V2 with sync per-element load replaced by cp.async
  2-stage double buffer. All math/masking/softmax/split/combine/output kept
  byte-identical → **V3 is bit-identical to V2** (torch.equal True everywhere).
  This was the explicit goal and the strongest possible correctness guarantee.
- **commit-group granularity = one 64-token tile** (4 K + 4 V physical-block
  sub-copies → one `cp_async_commit_group()`), `wait_group(1)` in steady state
  (depth-1 ping-pong), `wait_group(0)` in epilogue. Chosen over per-16-token-block
  groups to keep wait semantics simple; group depth never exceeded hardware limits.
- **2K + 2V SMEM (NOT B8-S's 2K+1V)**: reusing the depth that B6 validated;
  reducing V buffering shortens prefetch distance and (per the old B8-S negative
  result) raises long-scoreboard stalls. ncu confirms V3 long-scoreboard 20.9→0.86.
- **cp.async copy atom reuses B6's validated (4,8)×(1,8) tiling** for the
  swizzle(3,3,3) [8,64] atom — 128-bit vectors, `CopyG2SOp(GLOBAL)`. Not re-derived.
- **Invalid blocks never resolve block_table** (no OOB address); their SMEM rows are
  scalar zero-filled outside cp.async; partial tiles masked to -inf in QK, matching
  V2's empty-split neutral (partial_o=0, lse=-inf). Branches warp-uniform (no divergence).
- **Alignment domain enforced in runner, not silently fixed**: stride(-1)==1,
  data_ptr 16B aligned, block/head/slot strides multiples of 8 bf16. Padded-leading
  stride (slot-stride 136) still qualifies. Violation → ValueError (no clone/
  contiguous/sync/triton fallback).
- **Result**: 128K split=256 Stage-1 3.27ms→0.242ms (13.5x), DRAM 4.2%→57% peak
  (2.29 TB/s), no spill. Bottleneck moved from sync-load latency to DRAM bandwidth
  (the correct regime). Remaining gap vs continuous B7 (2.88 TB/s) = paged sub-copy
  granularity + stage depth; deeper pipeline / PDL deferred. V3 NOT committed, NOT
  wired to CUSTOM backend this round.

## D8. V3 evidence boundary & V3.1 measurement-integrity audit (pending)
Recording the honest scope of what the V3 logs actually prove, so the freeze commit
does not overclaim:
- **A. Log-backed** (`logs/v3_*.log`): V3==V2 bit-identical; correctness matrix PASS
  (2e-2 + strict 5e-3); benchmark `EXIT=0`; 128K split=256 e2e≈0.245ms; Nsight metric
  values; identity/shuffle penalties; no register spill.
- **B. Author-asserted only, NOT in any log** (not verified): physical GPU ("GPU 1/6"
  — logs only show `device: cuda:0`, no physical index / no `CUDA_VISIBLE_DEVICES`);
  the exact full command lines; the real **ncu shell exit codes** (ncu logs have no
  `EXIT=` sentinel).
- **C. NOT executed (no artifacts):** V3.1 two-GPU independent-process retest; V3.1
  Part VI length scaling; pointer rebinding / stale workspace audit; Nsight Systems
  timeline audit.
- Decision: freeze and commit V3 now (correctness/benchmark/Nsight evidence is solid);
  do NOT force-run V3.1 on the current shared machine (no clean GPU available). Items
  B and C are explicitly NOT PASS.

> V3.1 measurement-integrity audit remains pending and should be
> performed in the dedicated per-user container or another clean GPU
> environment. This does not invalidate the existing V3 correctness,
> benchmark, or Nsight evidence.
