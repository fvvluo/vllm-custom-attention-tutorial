# V2 — Direct Paged-KV warp-MMA Decode Prototype (Liu Xiaochen)

**V2 is an independent decode-only paged-KV prototype. It is NOT registered as
the active CUSTOM backend.** It is a standalone correctness/perf study that reads
the vLLM-style paged KV-cache directly (via `block_table`), using a warp-MMA
Tensor-Core kernel. Sync-load only (no cp.async / PDL yet — that is V3).

## Support domain (fixed target)
Decode-only: `q_len == 1` per sequence, `num_tokens == num_seqs` (packed in
request order). BF16, `D=128`, `Hq=64`, `Hkv=8` (GQA group 8), `block_size=16`,
SM90/H20, CUDA, contiguous or strided cache (stride-aware). Anything else raises
`TypeError`/`ValueError` (no silent fallback, no gather, no K/V copy).

## Paged addressing (in-kernel, no gather)
For logical token position `pos` of sequence `seq`, kv_head `kvh`, dim `d`:
```
logical_block  = pos // block_size          # block_size = 16
slot           = pos %  block_size
physical_block = block_table[seq, logical_block]
K = key_cache[physical_block, kvh, slot, d]  # read by real strides
V = value_cache[physical_block, kvh, slot, d]
```
A 64-token tile (`n_block=64`) spans 4 logical 16-token blocks, which may map to
non-contiguous physical blocks; SMEM rows 0..63 are filled in **logical** order.
Out-of-range tokens (`pos >= seq_len`) are zero-filled in SMEM and masked to
`-inf` in the score tile.

## MMA mapping
`warp.MmaF16BF16Op(bf16, Float32, (16,8,16))` — m16n8k16 HMMA, FP32 acc.
- Q tile `[16,128]`: 8 q_heads of this kv_head in rows 0..7, rows 8..15 zero pad.
- K tile `[64,128]`, score `[16,64]` FP32, P `[16,64]` BF16, V `[64,128]`, O `[16,128]` FP32.
- Both QK and PV run on Tensor Core; no SIMT dot/FMA main path.

## Split-KV + combine
Grid `[num_splits_max, kv_heads, num_seqs]`, one warp (32 threads) per
(split, kv_head, seq). `split_size_tokens` fixed; `valid_splits = ceil(seq_len/split_size)`,
`num_splits_max = ceil(max_seq_len/split_size)`; splits beyond a seq's valid range
write neutral (`lse=-inf`, `partial_o=0`). Each CTA processes `split_size/64`
tiles (e.g. split=256 → 4 tiles/CTA).
- Stage-1 partial: `partial_o [num_seqs,Hq,num_splits,D] FP32`, `partial_lse [num_seqs,Hq,num_splits] FP32`.
- Combine: grid `[Hq, num_seqs]`, 128 threads; `LSE-weighted` merge; writes BF16
  to `output[query_start_loc[seq], head, d]` (packed query token). Pure GPU.

## Correctness (rtol=atol=2e-2 tutorial tol; strict 5e-3-vs-ref diagnostic)
ALL PASS vs tutorial PyTorch reference AND tutorial Triton:
- basic seq_len ∈ {1,15,16,17,63,64,65,128,512} × seed {0,1,2026}
- tutorial decode [40,17,128]; irregular [1,17,65,257] / [127,128,129,1023]
- shuffled block_table; padded-leading-stride cache; explicit scale (default & 0.05)
- big: [40,512,8192], [131072], [8192,32768,131072] (shuffled) — max_abs ≤ 2.3e-4 vs ref.

## Microbenchmark (H20, warmup=10/iters=100/rounds=5, CUDA-event, same input;
tutorial Triton is the paged decode baseline)

### 128K single, split sweep (best = split 256)
| split | num_splits | Stage-1 ms | combine ms | end-to-end ms | speedup vs Triton | unique-KV(K+V) GB/s | workspace |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | **512** | 3.195 | 0.011 | **3.221** | **24.3x** | 167 | 16.9 MB |
| 512 | 256 | 3.513 | 0.007 | 3.530 | 22.2x | 152 | 8.5 MB |
| 1024 | 128 | 9.215 | 0.004 | 9.227 | 18.2x | 58 | 4.2 MB |

**Combine is ~0.3% of total; Stage-1 dominates (~99%).**

### earlier single-seq (cleaner GPU)
| seq_len | Triton ms | V2 e2e ms | speedup |
|---:|---:|---:|---:|
| 128 | 0.050 | 0.099 | 0.50x (fixed-overhead bound) |
| 512 | 0.192 | 0.171 | 1.12x |
| 8192 | 3.72 | 0.189 | 19.7x |
| 32768 | 19.6 | 1.07 | 18.3x |

### multi-sequence (seq_len=8192, split=256)
| num_seqs | end-to-end ms | speedup vs Triton | tokens/s |
|---:|---:|---:|---:|
| 1 | 0.189 | 19.7x | ~5300 |
| 4 | 2.340 | 4.19x | 1709 |
| 16 | 7.124 | 1.59x | 2246 |
V2 latency grows with num_seqs (more heavy warps, sync-load bound); Triton
parallelizes over more query tokens, so V2's relative lead shrinks.

### block_table identity vs randomized (same cache size)
- 8192: identity 0.418 ms vs shuffled 0.448 ms → **~7% penalty**.
- 32768: numbers contended on a shared GPU (identity run anomalously high);
  treated as indicative only. **We only claim block_table lookup is NOT a
  dominant bottleneck; we do not assert it is negligible without cleaner profiling.**

## Non-strict comparison to continuous B5/B7 (different repo/framework)
Under the unified K+V-bytes GB/s convention (128K K+V = 512 MiB):
- continuous B5 (sync) ≈ **1.81 TB/s**, B7 (cp.async+PDL) ≈ **2.88 TB/s**.
- V2 paged (sync) 128K ≈ **167 GB/s** — ~10-17x lower bandwidth utilization.
This is expected: V2 is a **synchronous-load** paged prototype. It far outpaces
the tutorial Triton (per-token serial-KV, ~24x at 128K), but is nowhere near the
HBM roofline because loads are not overlapped with compute.

## Bottleneck
Stage-1 synchronous paged K/V load — GB/s ≤167 (128K), far below H20 ~4 TB/s.
Combine negligible. block_table lookup not dominant (indicative). Latency-bound
on load, exactly the B5→B6 lesson.

## Next: V3 cp.async
Add cp.async double-buffered paged K/V staging (analogous to old B5→B6, ~1.5x
there) to overlap load with warp-MMA. V2 (this doc) is the frozen sync baseline.

## Limitations
Decode-only; fixed 64/8/128/16 shape; sync load; no cp.async/PDL; not wired to
CUSTOM backend; multi-seq scaling weak.

---

# V3 — cp.async Double-Buffered Paged-KV warp-MMA Decode (Liu Xiaochen)

**V3 is V2 with EXACTLY ONE variable changed:** the synchronous per-element paged
K/V load is replaced by a **cp.async 2-stage (2K+2V) double-buffered** pipeline
that prefetches the next 64-token logical tile while the warp-MMA consumes the
current one. Everything else — MMA atoms, QK/PV fragment mapping, tail masking,
online-softmax math, Split-KV schedule, empty-split neutral representation, packed
output, and the reused V2 combine — is byte-for-byte identical to V2. **V3 output
is bit-identical to V2 (`torch.equal` True) in every correctness case.** Still NOT
wired to the active CUSTOM backend.

Files: `paged_decode_stage1_v3.py`, `runner_v3.py`, `verify_paged_decode_v3.py`,
`bench_paged_decode_v3.py`, `ncu_stage1_probe.py`. Reuses `paged_decode_combine_v2.py`
(not copied). identity: `liuxiaochen-paged-decode-v3-mma-cpasync-2stage`.

## Evidence boundary (log-backed vs author-asserted vs pending)
The V3 numbers below are recorded honestly against what the logs actually contain.
Read this classification before citing any V3 result.

**A. Directly supported by existing logs** (`logs/v3_*.log`):
- V3 vs V2 **bit-identical** (`max_abs=0.000e+00`, `torch.equal`=True).
- Full **correctness matrix PASS** (tutorial 2e-2 + strict 5e-3-vs-ref).
- **Benchmark `EXIT=0`** on every bench segment.
- **128K split=256 V3 end-to-end ≈ 0.245 ms** (and the rest of the split/seqlen/multi-seq tables).
- **Nsight metric values** (HMMA count, DRAM throughput, long-scoreboard, tensor-pipe, regs, SMEM).
- **identity vs shuffle** block_table penalty numbers.
- **no register spill** (`local ld/st = 0`).

**B. Author-asserted only — NOT recorded in any log** (do not treat as verified):
- The physical GPU used (documented as "GPU 1/6") — logs only recorded `device: cuda:0`,
  and did not capture the physical index or `CUDA_VISIBLE_DEVICES`.
- The exact full `CUDA_VISIBLE_DEVICES=… python …` command lines.
- The real **shell exit code of the `ncu` runs** — the ncu logs end at the metrics table
  with no `EXIT=` sentinel.

**C. NOT executed (planned, no artifacts on disk):**
- V3.1 **two-GPU independent-process retest**.
- V3.1 **Part VI length scaling**.
- **pointer rebinding / stale workspace** dedicated audit.
- **Nsight Systems timeline** audit.

None of the items in B or C may be reported as PASS.

> V3.1 measurement-integrity audit remains pending and should be
> performed in the dedicated per-user container or another clean GPU
> environment. This does not invalidate the existing V3 correctness,
> benchmark, or Nsight evidence.

## Paged cp.async addressing
A 64-token logical tile = 4 logical KV blocks (`block_size=16`). For tile `t`:
```
tile_start   = split_start + t*64
logical_block_j = (tile_start//16) + j          # j = 0,1,2,3
physical_block_j = block_table[seq, logical_block_j]
cp.async key_cache[physical_block_j, kv_head, 0:16, 0:128] -> SMEM rows [16j:16j+16]
cp.async value_cache[physical_block_j, kv_head, 0:16, 0:128] -> SMEM rows [16j:16j+16]
```
All **4 K sub-copies + 4 V sub-copies of a tile form ONE cp.async commit group**
(`cp_async_commit_group()`), issued into logical row order (no gather, no
contiguous workspace, no physical-block sorting). A fully-invalid logical block
(`block_start_tok >= seq_len`) is NOT resolved through block_table (no OOB
address) and its SMEM rows are deterministically zero-filled; partially-valid
tiles are masked to `-inf` in QK, identical to V2. Branches are warp-uniform.

## Pipeline (prologue / steady / epilogue)
- **Prologue**: load Q once (scalar); prefetch tile 0 (8 sub-copies) → 1 commit group.
- **Steady** (tile t): if t+1 exists, prefetch tile t+1 into the other stage →
  `commit_group()` → `cp_async_wait_group(1)` (≤1 group in flight: tile t ready,
  t+1 loading) → `sync_warp` → QK→mask→softmax→PV on stage[t%2] → `sync_warp` → ping-pong.
- **Epilogue** (last tile): no prefetch; `cp_async_wait_group(0)`; compute; write partial_o/lse.

## SMEM layout (2K + 2V, matching validated B6; NOT B8-S's 2K+1V)
`sQ` single-buffered; `sK`/`sV` each 2 stages of `[64,128]` BF16 with
`swizzle(3,3,3)` (ldmatrix-compatible). Dynamic SMEM/CTA (ncu) = **69.63 KB**
(V2 = 36.86 KB); registers/thread = **181** (V2 = 168), **no spill** (local
ld/st = 0). cp.async copy atom = `CopyG2SOp(GLOBAL)`, 128-bit vectors,
thr_layout (4,8) × val (1,8) — reused from B6's validated tiling for this swizzle.

## cp.async alignment domain (runner raises, never silently fixes)
`_check_cpasync_alignment` requires `stride(-1)==1`, `data_ptr()` 16-byte aligned,
and block/head/slot strides multiples of 8 bf16 (16 B) so every token-row start is
16-byte aligned. Covers standard contiguous cache AND V2's padded-leading stride
(D+8 backing sliced to D → slot-stride 136, still 8-multiple). Violations raise
`ValueError` — no clone/contiguous/sync-fallback/triton-fallback.

## Correctness (V3 == V2 bit-identical; strict 5e-3-vs-ref; tutorial 2e-2)
ALL PASS, `V3==V2` = YES (max_abs 0.000e+00) in every case:
- basic {1,15,16,17,63,64,65,127,128,129,512} × seed {0,1,2026}
- tutorial [40,17,128]; irregular [1,17,65,257]/[127,128,129,1023]
- shuffle {0,1,2026}; padded-leading-stride; scale {default,0.05}
- big [40,512,8192], [131072], [8192,32768,131072] (shuffled).

## Microbenchmark (H20, warmup=10/iters=100/rounds=5, CUDA-event, same input, random order)

### 128K single, split sweep (best = split 256)
| split | nsm | V2 S1 ms | V3 S1 ms | V3 e2e ms | V3 vs V2 | V3 vs Triton | V3 GB/s (K+V) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | 512 | 3.194 | **0.220** | **0.245** | **13.15x** | 319.7x | 2193 |
| 512 | 256 | 3.508 | 0.371 | 0.383 | 9.22x | 204.6x | 1403 |
| 1024 | 128 | 4.192 | 0.578 | 0.578 | 7.27x | 135.4x | 929 |

### per-seq-len (num_seqs=1, split=256)
| seq_len | V2 S1 ms | V3 S1 ms | V3 e2e ms | V3 vs V2 | V3 vs Triton |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.086 | 0.011 | 0.098 | 0.99x (fixed-overhead) | 0.52x |
| 512 | 0.167 | 0.013 | 0.099 | 1.74x | 1.96x |
| 8192 | 0.180 | 0.025 | 0.098 | 1.90x | 31.3x |
| 32768 | 1.052 | 0.064 | 0.099 | 10.77x | 198.3x |
| 131072 | 3.194 | 0.220 | 0.245 | 13.15x | 319.6x |

### multi-sequence (seq_len=8192, split=256)
| num_seqs | V2 S1 ms | V3 S1 ms | V3 e2e ms | V3 vs V2 | V3 vs Triton | tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.180 | 0.025 | 0.098 | 1.90x | 31.3x | 10207 |
| 4 | 1.061 | 0.065 | 0.100 | 10.63x | 48.5x | 39790 |
| 16 | 3.203 | 0.219 | 0.231 | 13.92x | 22.2x | 69224 |
V3 scales far better than V2 across num_seqs (cp.async keeps the SMs fed as heavy
warps accumulate), so V3's lead over V2 grows with load rather than shrinking.

### block_table identity vs shuffle (V3 Stage-1)
<!-- "clean GPU" here is author-assertion; the run's physical GPU / CUDA_VISIBLE_DEVICES
     was not captured in the log (see Evidence boundary, item B). -->

| seq_len | identity S1 ms | shuffle S1 ms | penalty |
|---:|---:|---:|---:|
| 8192 | 0.0250 | 0.0250 | ~0% |
| 32768 | 0.0637 | 0.0643 | ~0.9% |
| 131072 | 0.2187 | 0.2197 | ~0.5% |
On a clean GPU the paged block_table permutation costs **<1%** for both V2 and V3
— block_table lookup is confirmed NOT a bottleneck (this supersedes V2's contended
32768 anomaly).

## Nsight (V2 vs V3 Stage-1, 128K split=256 shuffle)
<!-- "clean-ish GPU" is author-assertion; physical GPU / CUDA_VISIBLE_DEVICES and the
     ncu shell exit code were not captured in logs (see Evidence boundary, item B). -->

| metric | V2 sync | V3 cp.async | evidence |
|---|---:|---:|---|
| HMMA count | 2097152 | 2097152 | math unchanged |
| Stage-1 latency | 3.27 ms | 0.242 ms | 13.5x faster |
| DRAM throughput | 170 GB/s (4.2%) | 2.29 TB/s (56.9%) | load overlapped |
| long-scoreboard stall/issue | 20.90 | 0.86 | memory dependency gone |
| tensor pipe active | 0.73% | 10.12% | MMA fed |
| regs/thread | 168 | 181 | no spill (local ld/st = 0) |
| dyn SMEM/CTA | 36.86 KB | 69.63 KB | 2K+2V |
All 7 success-evidence checks pass: HMMA unchanged, cp.async present (DRAM 2.29 TB/s
+ long-scoreboard collapse), latency down, DRAM up, no spill, output unchanged.

## Bottleneck (V3) & next
V3 Stage-1 now runs at **~57% of DRAM peak (2.29 TB/s)** — memory-bandwidth bound,
which is the correct regime for decode. Remaining headroom vs continuous B7
(~2.88 TB/s) is the paged sub-copy granularity (4 physical blocks/tile vs one
contiguous block) and 2-stage depth. Next candidates (out of scope this round):
deeper cp.async pipeline (3-4 stage), PDL, or wiring into the CUSTOM backend.

## V3 limitations
Decode-only; fixed 64/8/128/16 shape; 2-stage depth; no PDL; not wired to CUSTOM
backend; requires 16-byte-aligned contiguous-D paged cache (raises otherwise).

## V3.1 measurement-integrity audit — RESULTS (executed on shared host)
Run with live per-step GPU rechecks and explicit `CUDA_VISIBLE_DEVICES` binding; the
physical GPU + UUID are recorded for every run. Only a test harness
(`verify_paged_decode_v3_integrity.py`) was added — no kernel/MMA/cp.async/combine
change, no PDL, no extra stage, not wired to CUSTOM backend. Full tables in
`docs/WORKLOG.md` ("Phase V3.1").

### Integrity audit — PASS (physical GPU 5, UUID 6afcc978-…-a586e)
`logs/v3_1_integrity.log` → `INTEGRITY_AUDIT=PASS`, `EXIT_CODE=0`, 26/26 checks
(seq_len=8192, split=256, shuffle): A→B→A, pointer rebinding (same compiled kernel
reused, no first-call capture), V3==V2 bit-identical (max_abs 0), V3 vs ref ≤5e-3
(2.3e-4 / 2.1e-4), K/V in-place data replacement, **partial_o NaN poisoning**,
output-pointer identity, empty-split partial_o==0 / lse==-inf.
- **Accurate limitation (not a failure):** the runner does `lse.fill_(-inf)` each call
  (runner_v3.py:136), so lse-poisoning is masked → **partial_o poisoning is the real
  stale-workspace evidence**. The integrity case has full/empty splits only; **partial
  tiles are covered by the V3 irregular correctness suite**, not here.

### Independent-process retest — cross-physical-GPU PASS
Two fresh processes, 128K/split=256, `--gpu 0` under `CUDA_VISIBLE_DEVICES`:
| process | phys GPU | UUID | V3 e2e ms | V3/V2 | GB/s(K+V) | exit |
|---|---|---|---:|---:|---:|---:|
| A | 5 | 6afcc978-…-a586e | 0.2445 | 13.193x | 2195 | 0 |
| B | 3 | 703ca4fb-…-b401d6 | 0.2457 | 13.081x | 2185 | 0 |
A/B diff ≤0.5% on e2e/bandwidth/speedup; both bit_identical, no OOM/CUDA-error/fallback.
**Reproduces the original 0.245 ms and 13.15x** on two distinct physical GPUs.

### Length scaling — PASS (physical GPU 3), all EXIT_CODE=0
| seq_len | num_splits | V3 S1 ms | V3 e2e ms | V3/V2 | GB/s(K+V) |
|---:|---:|---:|---:|---:|---:|
| 16384 | 64 | 0.0392 | 0.0991 | 7.20x | 677 |
| 32768 | 128 | 0.0646 | 0.1005 | 10.68x | 1335 |
| 65536 | 256 | 0.1146 | 0.1312 | 13.58x | 2047 |
| 131072 | 512 | 0.4656 | 0.5382 | 13.17x | 997 |
V3 Stage-1 rises monotonically (no missing kernel); combine stays ~0.006–0.011 ms.
**128K point is CONTENDED** (a foreign task co-landed on GPU 3 mid-sweep → all three
kernels ~2.2x slower, V3/V2 ratio unchanged at 13.17x). The clean 128K numbers are the
independent-process retest above (0.245/0.246 ms, ~2190 GB/s).

### CUDA Event timing audit — sound (read-only)
Same CUDA stream; Stage-1 + combine both inside the `start.record()…end.record()`
window (runner_v3.py:182–183); JIT/allocation excluded; **no CUDA Graph**; randomized
order cannot misalign events; Stage-1-only vs end-to-end are separate series.

### Still pending
- **Nsight Systems timeline audit** — not run this round.
This round is audit-only: V3 is NOT committed/pushed for V3.1.

---

# V4 — minimal vLLM CUSTOM-backend integration (route supported decode to V3)

V4 wires V3 into the live CUSTOM backend WITHOUT changing any V3/V2 kernel math
(kernel diff = 0 bytes), no PDL, no extra stage, no new backend, no Prefill kernel.
Files: `vllm_adapter_v4.py` (dispatch), `verify_v4_adapter.py` (synthetic gate),
`verify_v4_backend_forward.py` (real-forward gate). `custom_triton_backend.py` gains
`max_query_len`/`max_seq_len` in its metadata/builder and reroutes ONLY the attention
read to V3 when the batch is strictly in the V3 support domain.

## Dispatch
`CustomTritonImpl.forward` always writes the KV cache
(`triton_reshape_and_cache_flash`), then calls `try_v3_decode(...)`. If
`can_use_v3_decode(...)` (a pure predicate: pure decode `max_query_len==1`, bf16,
Hq64/Hkv8/D128/bs16, SM90, causal, finite positive scale, 16B-aligned
last-dim-contiguous cache) returns True AND the flag is on, V3 runs in place on the
LIVE tensors (zero-copy); otherwise the existing tutorial-Triton path runs. Any V3
runtime error disables V3 for the process and falls back (never crashes the server).

## Feature flag
`LIUXIAOCHEN_PAGED_DECODE_V3`: `1` enables V3 inside the support domain; `0`/unset
keeps the pure Triton baseline. Debug totals via `LIUXIAOCHEN_PAGED_DECODE_V3_DEBUG`.

## Evidence boundary
- **PASS (synthetic):** `logs/v4_synthetic_gpu1.log` (GPU 1, EXIT_CODE=0, 37/37) —
  adapter-V3 == direct-V3 bit-identical on the packed split-view layout, vs PyTorch
  ref ≤5e-3, output in place, A→B→A stable, 7 fallback categories correct, no silent clone.
- **Resource-blocked (NOT code):** Qwen3-32B BF16 service could not start (shared-GPU
  contention during model load). Real service smoke, service-level decode-hit / prefill-
  fallback evidence, and TTFT/TPOT/E2E remain **pending** — synthetic PASS is not a service PASS.
