# WORKLOG

## Phase V1 — environment setup & baseline reproduction (Liu Xiaochen)

### Done
1. Created personal root `/dockerdata/liuxiaochen`; confirmed 2.6 TB free, empty target paths.
2. Cloned tutorial repo → `/dockerdata/liuxiaochen/vllm_tutorial` @ `002b805` (main); saved to `/tmp/liuxiaochen_vllm_tutorial_initial_commit.txt`.
3. Audited all hardcoded paths (author `landojiang`) — all scripts env-overridable
   (`${VLLM_SRC:-...}` etc.). setup script only clones + symlinks .so; no delete /
   pip / compile / bashrc writes. serve scripts default GPU0/PORT8000 (overridden).
4. Verified env: /usr/bin/python 3.12.13, torch 2.13.0+cu130, CUDA 13.0, nvcc 13.0,
   8× H20 CC9.0; installed vLLM 0.18.1rc1.dev3933+ga49d37c6b; model Qwen3-32B 62 GB.
   All match tutorial spec.
5. Wrote `scripts/setup_vllm_source_liuxiaochen.sh` (path params + guards: refuse
   dirty/non-empty non-repo VLLM_SRC, verify commit, no uninstall/recompile).
   `bash -n` OK.
6. Ran it: vLLM src @ a49d37c6b, 16 .so + _version.py symlinked, 0 broken links,
   `vllm.__file__` from source tree, `_C_stable_libtorch` OK.
7. `pip install -e .` → entry point `custom_triton -> custom_backend.plugin:register`;
   `custom_backend` imports from tutorial dir.
8. Wrote `serve_qwen3_flashattn_liuxiaochen.sh` + `serve_qwen3_custom_liuxiaochen.sh`
   (mandatory GPU/PORT, safety checks, personal PYTHONPATH+logs). `bash -n` OK.
9. Part 1 FlashAttention baseline: GPU4/8401 MAX_LEN=8192 → backend flash_attn,
   startup complete, smoke → 42 PASS. Stopped cleanly (port+GPU released).
10. Part 2 CUSTOM backend: GPU4/8402 → "Using AttentionBackendEnum.CUSTOM backend",
    smoke → 42 PASS. Stopped cleanly.
11. Part 3 paged-attn correctness (GPU4): prefill/decode/mixed ALL PASS
    (7.81e-3 / 3.89e-3 / 7.75e-3, bf16 tol 2e-2, unchanged).
12. Audited `paged_attention_triton` interface (see DECISIONS.md).
13. Added profiler/humaneval-result ignores to personal `.gitignore`.
14. Wrote HANDOFF.md / WORKLOG.md / DECISIONS.md.

### NOT done (out of scope this phase)
- No B7 port; no kernel edits; no perf optimization; no 100k perf test; no YaRN
  128k; no HumanEval; no Nsight; old attention-test repo untouched.

### GPU/port discipline
Used GPU4 (idle, ~0% util, no other user process) for all runs; ports 8401/8402,
both released after use. No processes killed except my own serve (SIGINT).

## Phase V2 / V2.1 — paged-KV warp-MMA decode prototype (Liu Xiaochen)

- New independent dir `custom_backend/liuxiaochen_paged_decode/` (stage1/combine/
  runner/verify/bench/README). Decode-only, NOT registered as CUSTOM backend.
- Correctness ALL PASS (tutorial tol 2e-2 + strict 5e-3-vs-ref): basic seq sweep,
  tutorial decode, irregular, shuffled block_table, padded stride, explicit scale,
  big [40,512,8192]/[131072]/[8192,32768,131072]. max_abs ≤ 2.3e-4 vs PyTorch ref.
- 128K single best split = 256: end-to-end 3.22ms, 24.3x vs tutorial Triton;
  Stage-1 3.195ms (~99%), combine 0.011ms (~0.3%). split 512=3.53ms, 1024=9.23ms.
- single-seq (cleaner GPU): 8192=0.189ms/19.7x, 32768=1.07ms/18.3x, 512=0.171ms/1.12x,
  128=0.099ms/0.50x (fixed-overhead bound).
- multi-seq 8192: ns=4 2.34ms/4.19x, ns=16 7.12ms/1.59x (V2 latency grows; Triton
  parallelizes over tokens so relative lead shrinks).
- block_table identity vs shuffled @8192: 0.418 vs 0.448ms → ~7% penalty
  (indicative; 32768 pair contended, not conclusive). Only claim: not dominant.
- unified K+V GB/s: V2 128K ≈167 GB/s vs continuous B5≈1.81TB/s, B7≈2.88TB/s.
  V2 is sync-load; bottleneck = synchronous paged K/V load (latency-bound).
- Logs: v2_verify_basic/suites/big.log, v2_bench_small/128k_split_sweep/multiseq/blocktable.log
- V2 frozen as the sync baseline. Next: V3 cp.async double-buffer (overlap load).

## Phase V3 — cp.async double-buffered paged-KV warp-MMA decode (Liu Xiaochen)

- ONE variable changed vs V2: sync per-element paged load → **cp.async 2-stage
  (2K+2V) double buffer**. New files stage1_v3/runner_v3/verify_v3/bench_v3 +
  ncu_stage1_probe; reuses combine_v2 (not copied). V2 files unchanged (git diff empty).
  identity `liuxiaochen-paged-decode-v3-mma-cpasync-2stage`.
- cp.async addressing: 64-token tile = 4 logical blocks; each resolved via
  block_table to a physical block, whole [16,128] issued cp.async → SMEM rows
  [16j:16j+16] in logical order. 4 K + 4 V sub-copies = ONE commit group.
  Fully-invalid block never touches block_table (no OOB), zero-filled; partial tile
  masked to -inf in QK. Prologue prefetch tile0; steady: prefetch t+1 →
  commit → wait_group(1) → compute t → ping-pong; epilogue wait_group(0).
- Correctness: **V3 == V2 bit-identical (torch.equal True, max_abs 0.000e+00) in
  EVERY case** — basic {1..512}×seed{0,1,2026}, tutorial, irregular, shuffle{0,1,2026},
  padded-stride, scale, big [40,512,8192]/[131072]/[8192,32768,131072]. strict 5e-3
  + tutorial 2e-2 all PASS.
- Perf (H20, same input, random order). **128K split=256: V3 0.245ms e2e vs V2
  3.221ms = 13.15x, 319x vs Triton, 2193 GB/s (K+V).** Split sweep best=256
  (512=0.383ms/9.2x, 1024=0.578ms/7.3x). Per-seq: 8192 1.90x, 32768 10.8x, 131072
  13.15x over V2. Multi-seq 8192: ns4 10.6x, ns16 13.9x over V2 (V3 scales UP with
  load, opposite of V2). combine still ~0.3%.
- Gate: required 128K split=256 < 3.221ms and ≥15% (≤2.74ms). **Achieved 0.245ms
  — far beyond (13.15x).**
- block_table identity vs shuffle: 8192 ~0%, 32768 ~0.9%, 131072 ~0.5%
  penalty — confirms lookup <1%, NOT a bottleneck (supersedes V2's contended 32768).
- Nsight (V2 vs V3 Stage-1, 128K split=256): HMMA 2097152 unchanged; latency
  3.27ms→0.242ms; DRAM 170 GB/s→2.29 TB/s (57% peak); long-scoreboard stall/issue
  20.90→0.86; tensor pipe 0.73%→10.12%; regs 168→181, NO spill (local ld/st 0);
  dyn SMEM 36.86KB→69.63KB (2K+2V). All 7 success-evidence items pass.
- Bottleneck now: DRAM-bandwidth bound at ~57% peak (correct decode regime).
  Headroom vs continuous B7 (~2.88 TB/s) = paged sub-copy granularity + 2-stage depth.
- Logs: v3_verify_all.log, v3_verify_big.log, v3_bench_128k_split_sweep.log,
  v3_bench_seqlens.log, v3_bench_blocktable.log, v3_ncu_v2.log, v3_ncu_v3_full.log.
- NOT committed/pushed. NOT wired to CUSTOM backend. Archive
  /tmp/liuxiaochen_before_v3_cpasync.tar.gz (sha256 6a7d3ace…5b9bd).

### V3 evidence boundary (honest scope of proof)
- **A. Directly supported by logs** (`logs/v3_*.log`): V3==V2 bit-identical
  (max_abs 0.000e+00); correctness matrix PASS (2e-2 + strict 5e-3); benchmark
  `EXIT=0` on every segment; 128K split=256 V3 e2e≈0.245ms; Nsight metric values;
  identity/shuffle penalties; no register spill.
- **B. Author-asserted only, NOT recorded in any log** (not verified): the physical
  GPU used (worklog says "idle GPU 1/6" — logs only recorded `device: cuda:0`, with
  no physical index and no `CUDA_VISIBLE_DEVICES`); the exact full command lines;
  the real **shell exit code of the `ncu` runs** (ncu logs end at the metrics table,
  no `EXIT=` sentinel).
- **C. NOT executed — planned, zero artifacts on disk:** V3.1 two-GPU
  independent-process retest; V3.1 Part VI length scaling; pointer rebinding /
  stale workspace dedicated audit; Nsight Systems timeline audit.
- None of the B or C items are PASS and must not be recorded as such.

> V3.1 measurement-integrity audit remains pending and should be
> performed in the dedicated per-user container or another clean GPU
> environment. This does not invalidate the existing V3 correctness,
> benchmark, or Nsight evidence.

- GPU discipline: intended to use an idle GPU; runtime device index was not
  captured in the logs (see item B).

## Phase V3.1 — measurement-integrity audit + independent-process retest (Liu Xiaochen)

Executed on the shared host (NOT a dedicated container); every step re-checked the
GPU live and bound the physical device via `CUDA_VISIBLE_DEVICES` (recorded below).
No V3/V2 kernel math / MMA / cp.async / combine changed; no PDL; no extra stage; not
wired to CUSTOM backend. New file only: `verify_paged_decode_v3_integrity.py`.

### A. Integrity audit — PASS (physical GPU 5, UUID 6afcc978-…-a586e)
`logs/v3_1_integrity.log`, `INTEGRITY_AUDIT=PASS`, `EXIT_CODE=0`, 26/26 checks PASS
(seq_len=8192, num_seqs=1, split=256, shuffle):
- A→B→A: A1==A2 bit-identical, A≠B (max_abs 9.84e-2) — PASS
- pointer rebinding: same compiled kernel reused for B and A2 (cache size 1), no
  first-call tensor capture — PASS
- V3 vs V2: torch.equal, max_abs 0.000e+00 (both A and B) — PASS
- V3 vs PyTorch ref: A 2.32e-4, B 2.12e-4 (≤5e-3) — PASS
- K/V data replacement (same object, in-place swap): output changes then restores
  to A1; data_ptr stable — PASS
- partial_o NaN poisoning: output == A1, no NaN/Inf — PASS (**the real
  stale-workspace evidence**)
- output pointer: outA/outB data_ptr unchanged across calls — PASS
- empty-split neutrality (max_seq_len 10240 > 8192): empty partial_o==0,
  empty lse==-inf, valid splits NaN-free, output==A1 — PASS
- **Accurate limitation (NOT a failure):** the runner unconditionally runs
  `lse.fill_(-inf)` each call (runner_v3.py:136), so lse-poisoning is wiped before
  the kernel and is NOT independent stale-workspace evidence; partial_o poisoning is
  the load-bearing probe. The integrity case (seq 8192, split 256) has full/empty
  splits only and does NOT cover partial tiles — partial tiles are covered by the V3
  irregular correctness suite (verify_paged_decode_v3.py).

### B. Independent-process retest — cross-physical-GPU PASS
Two fresh Python processes, sequential, 128K/split=256/warmup10/iters100/rounds5,
`--gpu 0` inside each (physical GPU chosen via `CUDA_VISIBLE_DEVICES`):
- **Process A — physical GPU 5** (UUID 6afcc978-…-a586e), `logs/v3_1_retest_process_a_gpu5.log`, EXIT_CODE=0:
  V2 S1 3.2006ms; V2 e2e 3.2261 (min3.2258/max3.2264/p90 3.2263); V3 S1 0.2197ms;
  V3 e2e 0.2445 (min0.2445/max0.2446/p90 0.2446); combine 0.0105ms; V3/V2 13.193x;
  V3/Triton 320.11x; 2195 GB/s (K+V); workspace 16908288 B; bit_identical=True; no OOM/CUDA-error/fallback.
- **Process B — physical GPU 3** (UUID 703ca4fb-…-b401d6), `logs/v3_1_retest_process_b_gpu3.log`, EXIT_CODE=0:
  V2 S1 3.1883ms; V2 e2e 3.2145 (min3.2142/max3.2155/p90 3.2145); V3 S1 0.2202ms;
  V3 e2e 0.2457 (min0.2456/max0.2459/p90 0.2458); combine 0.0107ms; V3/V2 13.081x;
  V3/Triton 318.91x; 2185 GB/s (K+V); bit_identical=True; no OOM/CUDA-error/fallback.
- **A vs B**: V3 e2e diff 0.49%, V2 e2e diff 0.36%, bandwidth diff 0.46%, speedup
  13.19x vs 13.08x — all ≤10% (excellent). Both cards keep V3 ≫ V2.
- **Reproduces original V3 0.245ms and 13.15x** (A 0.2445/13.19x, B 0.2457/13.08x).

### C. Length scaling — PASS (physical GPU 3), `logs/v3_1_length_scaling_gpu3.log`, all EXIT_CODE=0
num_seqs=1, split=256, warmup10/iters100/rounds5.
| seq_len | num_splits | V2 S1 ms | V2 e2e ms | V3 S1 ms | V3 e2e ms | combine ms | V3/V2 | GB/s(K+V) | V3 e2e min/med/max/p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16384  | 64  | 0.7062 | 0.7136 | 0.0392 | 0.0991 | 0.0061 | 7.20x  | 677  | 0.0971/0.0991/0.1083/0.0992 |
| 32768  | 128 | 1.0633 | 1.0735 | 0.0646 | 0.1005 | 0.0062 | 10.68x | 1335 | 0.1002/0.1005/0.1021/0.1006 |
| 65536  | 256 | 1.7649 | 1.7809 | 0.1146 | 0.1312 | 0.0065 | 13.58x | 2047 | 0.1310/0.1312/0.1312/0.1312 |
| 131072 | 512 | 7.0407 | 7.0886 | 0.4656 | 0.5382 | 0.0107 | 13.17x | 997  | 0.5380/0.5382/0.5384/0.5384 |
- Trend: V3 Stage-1 rises monotonically 0.039→0.065→0.115→0.466ms (not flat — no
  missing kernel); 64K→128K Stage-1 ~4.06x (work ~2x per doubling; extra factor is
  the 128K contention below). combine stays ~0.006–0.011ms (never the bottleneck).
- **128K length-scaling point is CONTENDED/anomalous:** midway through the sweep a
  foreign task (PID 1804858, 92GB, 100% util) co-landed on GPU 3. All three kernels
  slowed ~2.2x uniformly at 128K (triton 78→168ms, V2 3.23→7.09ms, V3 0.245→0.538ms)
  while V3/V2 held at 13.17x — a bandwidth-contention signature, NOT a kernel change.
  The clean 128K numbers are Process A/B (0.245/0.246ms, ~2190 GB/s). 16K/32K/64K ran
  clean; GB/s climbs 677→1335→2047 into the memory-bound regime as expected.

### D. CUDA Event timing audit (read-only source review) — sound
`bench_paged_decode_v3.py time_fn` (L73–82): warmup loop L74–75 → `synchronize` L76 →
`s.record()` L78 (before the iters loop) → iters×`fn()` L79–80 → `e.record()`+`synchronize`
L81 → `elapsed_time` L82. For end-to-end V3, `fn=run_v3` (L121–125) → `paged_decode_v3`
launches Stage-1 (runner_v3.py:182) AND combine (runner_v3.py:183) on the SAME stream
(runner_v3.py:155–156) — both inside the event window. Stage-1-only (`s1v3`=run_stage1,
runner_v3.py:246–248) is a separate series from e2e (not conflated). JIT+alloc excluded
(compiled/warmed at L140 before timing; workspace cached). `random.shuffle` (L151) only
permutes which self-contained `time_fn` runs — cannot misalign events. **No CUDA Graph**
(grep: zero graph/capture references). No missing-kernel risk for e2e.

### E. Still pending
- **Nsight Systems timeline audit** — not run this round (kept pending).
- No cross-GPU item is pending: A (GPU5) + B (GPU3) both completed.

### V3.1 environment notes / GPU interference this round
Shared host, not exclusive. GPU 5 clean at Process-A start (47MiB, 0%, 38°C); a
foreign 88GB task landed on GPU 5 right after A finished. GPU 3 clean for Process B
and for 16K/32K/64K, then contended at the 128K length-scaling point (see C). No
process was killed; no card was auto-switched to stitch a single result set.
- Logs: v3_1_integrity.log, v3_1_retest_process_a_gpu5.log,
  v3_1_retest_process_b_gpu3.log, v3_1_length_scaling_gpu3.log.
- Historical stub (NOT this round's evidence): v3_1_integrity_import_failure_20260729_1912.log.

## Phase V4 — minimal vLLM CUSTOM-backend integration of V3 paged decode (Liu Xiaochen)

Goal: route STRICTLY-supported pure-decode CUSTOM attention calls to the V3
cp.async paged-KV kernel; keep the existing tutorial-Triton path for everything
else. NO kernel/MMA/cp.async/combine change (kernel diff = 0 bytes), no PDL, no
extra stage, not a new backend, no Prefill kernel.

New files: `vllm_adapter_v4.py` (dispatch), `verify_v4_adapter.py` (synthetic
gate), `verify_v4_backend_forward.py` (real-forward gate). Modified:
`custom_triton_backend.py` (+`max_query_len`/`max_seq_len` in metadata/builder;
reroute only the READ to V3 when supported). Feature flag
`LIUXIAOCHEN_PAGED_DECODE_V3` (1=on inside support domain, 0/unset=Triton baseline).

### A. Directly verified (real, PASS)
- Backend API + read-path audit: registration `plugin.py:register` →
  `CustomTritonBackend`; forward `CustomTritonImpl.forward` writes KV via
  `triton_reshape_and_cache_flash` then reads via `paged_attention_triton`.
- Packed KV-cache layout `(num_blocks,Hkv,block_size,2*hs)` split into K/V half-views
  → `stride(-1)==1`, slot-stride 2*hs, value offset +hs — exactly V3's cp.async
  alignment domain, so V3 consumes the LIVE views zero-copy (no clone/contiguous/gather).
- Feature flag on/off honored; `can_use_v3_decode` is a pure predicate (no alloc/
  sync/mutation); dispatch evidence is low-frequency (first-hit + first-of-reason +
  atexit totals).
- **Synthetic adapter gate — PASS** (`logs/v4_synthetic_gpu1.log`, physical GPU 1,
  EXIT_CODE=0, 37/37): adapter-V3 == direct-V3 **bit-identical** (max_abs 0) on the
  packed split-view layout; V3 vs PyTorch reference ≤5e-3; output written in place
  (data_ptr/shape/dtype OK); A→B→A stable (no first-tensor capture); seven fallback
  categories (prefill / mixed / fp16 / head_dim / block_size / misaligned-stride /
  non-causal) correctly rejected with reasons, no silent clone. Totals v3_hits=6.
- V2/V3 kernel files unchanged (git diff = 0 bytes).

### B. Attempted but resource-blocked (NOT a code failure)
- Three Qwen3-32B BF16 engine-init attempts (GPU 1 ×1, GPU 2 ×2) all failed at
  startup with `ValueError: Free memory on device cuda:0 (30.31 / 5.98 GiB) < desired
  GPU memory utilization (0.9, 85.5 GiB)`. Shared host runs 6+ other Qwen3-32B servers
  (ports 8000/8004/8005/8007/8014/8020) that seize any freed GPU within the ~30s
  model-load window. These are NOT adapter/V3 CUDA errors; no service-level V3-hit
  evidence was produced.

### C. Not yet verified (service level)
- Real service smoke (baseline + V4), service-level prefill fallback + decode hit,
  TTFT/TPOT/E2E, HumanEval. These remain pending; do NOT read the synthetic PASS as a
  service PASS.
