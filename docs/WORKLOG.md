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

### D (V4-R). Real backend-forward gate — PASS; one BF16 service attempt — RESOURCE_CONTENDED
- **Real CustomTritonImpl.forward gate — PASS** (`logs/v4_backend_forward_gpu5.log`,
  physical GPU 5 UUID 6afcc978, EXIT_CODE=0, `BACKEND_FORWARD_GATE=PASS`, 24/24). Drives
  the actual forward (real metadata + real KV-cache WRITE via reshape_and_cache + real
  read dispatch); only fixture is a tiny `_Layer` exposing `_k_scale`/`_v_scale`=1.0.
  - [A] flag=0 → Triton read path, vs ref 2.17e-4, V3 hits=0.
  - [B] flag=1 → forward HITS V3 (hit +1 each of ns1_sl128 / ns1_sl8192 / ns4_mixed);
    forward(V3) == direct V3 **bit-identical** (max_abs 0); vs ref ≤5e-3; output
    `[num_tokens,num_heads,head_size]` bf16 contract OK.
  - [C] prefill (max_query_len>1) → Triton fallback (V3 hits unchanged, reason
    `not_pure_decode`), vs causal ref within tutorial 2e-2 (7.76e-3; strict-5e-3 is too
    tight for the bf16 Triton path — fixture tolerance, not a code fault).
  - [D] fp16 decode → fallback (reason `query_dtype=torch.float16`), correct.
  - [E] A→B→A through real forward: A1==A2, A≠B (no first-tensor capture).
  - Totals: v3_hits=6, fallbacks=2.
- **BF16 service smoke — RESOURCE_CONTENDED (NOT a code failure):** one allowed attempt,
  baseline flag=0 on physical GPU 6 (UUID eda292b3), GPU_MEM_UTIL=0.78, MAX_LEN=8192,
  port 8403. Passed the startup free-mem check and loaded to "Using
  AttentionBackendEnum.CUSTOM backend", then `torch.OutOfMemoryError` mid weight-load:
  two foreign processes (32.8 GiB + 62.1 GiB) co-landed on GPU 6 during the ~20s load
  (`logs/v4_serve_baseline_gpu6_8403.log`). Per discipline: no 2nd BF16 attempt, no
  card-swap, no FP8 (no repo FP8 evidence — grep empty). Service-level V3-hit /
  prefill-fallback / TTFT/TPOT/E2E remain PENDING on a durably-free ≥75 GiB GPU.
- Min-viable BF16 (from prior successful serve log): weights+non-torch ≈ 61.9 GiB,
  peak activation ≈ 1.48 GiB; needs startup free ≈ util×95 GiB. util 0.78 ⇒ ~74 GiB
  free needed, leaving ~10 GiB KV (~40k tokens) — ample for an 8192-len smoke.

## Phase V4-R Final Sprint — legal refactor + live service (official 100K NOT completed)

### Submission-legal refactor (obey official README hard rule)
The official README (origin/main) states the ONLY pre-existing file we may modify is
`custom_backend/triton_attention.py`. The earlier V4 edited `custom_triton_backend.py`
(forbidden). Refactor: **reverted `custom_triton_backend.py` to pristine (002b805)**;
moved the V3 dispatch INTO `paged_attention_triton(...)` in `triton_attention.py` (legal
file). Pure-decode is now detected from the live tensors (num_tokens==num_seqs and
query_start_loc==[0,1,...,N]); `max_seq_len` derived from `seq_lens.max()` on the decode
path only. New adapter entry `try_v3_decode_from_triton_args`. Re-ran the real
backend-forward gate: **24/24 PASS, EXIT_CODE=0** (`logs/v4_backend_forward_legal_gpu0.log`,
GPU 0) — forward hits V3, bit-identical to direct V3, prefill/fp16 fall back.

### Live Qwen3-32B V4 service — READY + smoke PASS + V3 dispatch CONFIRMED (GPU 5, UUID 6afcc978)
`scripts/run_final_v4_evaluation.sh` (new), YaRN HF_OVERRIDES (factor 4.0), MAX_LEN 102400,
gpu-memory-utilization 0.95, `--attention-backend CUSTOM`, `LIUXIAOCHEN_PAGED_DECODE_V3=1`.
- Engine init clean: KV cache 26.84 GiB / 109,952 tokens; **Application startup complete**.
- **Smoke 42 = PASS**; 64-token generation PASS (req1=77, req2=44 tokens, coherent).
- **FIRST V3 decode HIT confirmed** in serve log; **Prefill fallback confirmed**
  (`not_pure_decode(num_tokens!=num_seqs)`). No OOM/CUDA-error/runtime-disable at smoke.

### Official 100K E2E — NOT COMPLETED (perf exit=1)
The official `perf_test.py` (input 100000 / output 64) **warmup request dropped**:
`RemoteProtocolError: peer closed connection without sending complete message body`.
Actual tokenized input ~95,653. No valid V4 E2E was produced → cannot compare against the
official baseline (147.0 s = flash_attn, single H20, `E2E = TTFT + 1000×TPOT`). Root
architectural cause: the CUSTOM backend keeps the teaching **Triton prefill**, which the
repo itself notes cannot practically run 100K; the long prefill stalled/reset the stream.
V3 only accelerates DECODE; the score adds TTFT linearly, so a slow 100K prefill dominates.

### HumanEval — 0/164 via harness (NOT a valid correctness credential)
`humaneval_generate.py` returned `    pass\n` for every task → pass@1 = 0/164. This is a
generation-harness/prompt-format artifact (the SAME live service answered "42" and wrote
coherent poems correctly), NOT V3 kernel garbage — but without a real pass@1 we do NOT
claim correctness. HumanEval remains PENDING.

### Verdict
V4 code path fully validated offline (synthetic 37/37, backend-forward 24/24) AND live
(service ready, 42, V3 decode HIT, prefill fallback). The OFFICIAL 100K E2E and HumanEval
pass@1 were NOT completed (100K prefill stream reset + HumanEval harness returned stubs).
Therefore the final performance claim is **NOT PASS** — recorded honestly, no fabricated
speedup. The 13.1x is a decode-kernel microbenchmark and is NOT a service-level result.

## Phase V4-R Final Rescue — VALID official 100K result obtained (FAIL: 201.0s > 147.0s)

### 100K reset root cause (Checkpoint R1): A. SERVER_ALIVE_CLIENT_RESET + glacial Triton prefill
Prior run's serve log ended idle with NO error/OOM/CUDA/crash (only the cleanup
resource_tracker warning); the same server then answered all 164 HumanEval requests →
server never died. The real blocker was prompt throughput **2.4–5.1 tok/s** on the
teaching Triton prefill → ~95,653-token prefill ≈ 8+ h → client `RemoteProtocolError`.
Chunked prefill was ALREADY on (`max_num_batched_tokens=8192`) and did not help.
dmesg Xid faults were all foreign PIDs. So the Triton prefill is unusable at 100K, but
this was NOT a kernel crash.

### Legal fast-prefill hybrid (Checkpoint R2)
Per the official README, only `triton_attention.py` is editable but NEW files + imports
called inside `paged_attention_triton` are allowed. Added `vllm_prefill_fa.py`: the
prefill/mixed path now calls vLLM's own paged FlashAttention (`flash_attn_varlen_func`
with `block_table=`) on the SAME packed KV-cache (K/V half-views transposed HND→NHD, a
view, no copy/gather). Architecture: **Prefill→paged FlashAttention, Decode→V3,
else→Triton**. Synthetic gate `verify_fa_prefill.py`: **6/6 PASS** (vs PyTorch ref ≤1e-2).

### Official 100K + 64 (Checkpoint R4) — VALID, 3/3 runs, physical GPU 3 (UUID 703ca4fb)
`logs/v4_final_perf_gpu3_20260730_105800.log`, perf exit=0, official defaults
(input 100000→~95,653 tokens, output 64, baseline 147.0):
| run | TTFT (s) | decode tok/s |
|---|---:|---:|
| 1 | 153.620 | 21.2 |
| 2 | 153.833 | 21.0 |
| 3 | 153.473 | 21.1 |
- median TTFT = **153.620 s**, TPOT = **47.42 ms/tok**, decode 21.1 tok/s.
- **E2E score = TTFT + 1000×TPOT = 201.042 s**; baseline 147.0 → **speedup 0.7312x (SLOWER)**.
- Live smoke PASS (42, 64-token gen); V3 decode HIT + prefill fallback confirmed; no
  OOM/CUDA-error/runtime-disable; server stayed alive; no GPU contention on GPU 3.
- FA prefill compute is actually fast (serve log burst `prompt throughput 9569.6 tok/s`),
  but single-request 100K TTFT is still ~153 s end-to-end (chunked-prefill scheduling +
  client tokenization of the 95k prompt), and **TTFT alone (153.6 s) already exceeds the
  147 s baseline** — since the score adds TTFT linearly, no decode speedup (even perfect
  0 ms) can win. TPOT 47 ms also indicates decode is not V3-dominated at this scale.

### Verdict: FAIL (valid official test completed)
V4 produced a **valid, honest official 100K E2E = 201.0 s**, which is **> 147.0 s**, so
the performance target is NOT met. The decode kernel's 13.1x microbenchmark does NOT
translate to a service win because 100K prefill TTFT dominates the score and exceeds the
baseline on its own. HumanEval was not scored (rules: only if score<147). No fabricated
numbers. The fast-prefill hybrid is kept because it is correct and a genuine improvement
(100K prefill now completes vs the prior unusable Triton stall), documented as not beating
baseline.

## Phase V4-R Final Diagnostic — service bottleneck located (config, not kernel/dispatch)

Goal: find a service-level lever to drop the official score below 147.0 s. Result: the
bottleneck is precisely attributed, but the fix is illegal under the grading rules.

### Same-GPU A/B diagnostic (authoritative)
One flash_attn baseline run on the SAME config (Qwen3-32B bf16, max_len 102400, util
0.95, chunked prefill, YaRN, input 100000/output 64) — only `--attention-backend`
differs (and baseline does NOT use `--enforce-eager`/`--no-async-scheduling`):
| metric | baseline flash_attn (GPU6) | V4 CUSTOM (GPU3) | gap |
|---|---:|---:|---:|
| TTFT | 111.440 s | 153.620 s | +42.2 s |
| TPOT | 35.54 ms | 47.42 ms | +11.9 ms |
| **E2E score** | **146.979 s** (=official 147.0, 1.0001x) | **201.042 s** | +54 s |
Baseline reproduced the official 147.0 s exactly → measurement validated.

### BASELINE_V4_CONFIG_DIFF (sorted by impact)
1. **CUDA Graph: baseline ON (FA3) vs V4 OFF** — HIGH (TPOT + per-step overhead).
2. **enforce_eager: baseline NO vs V4 YES** — HIGH.
3. **async_scheduling: baseline ON vs V4 OFF** — MED.
All three stem from ONE cause: the CUSTOM backend declares `_cudagraph_support=NEVER`
in `custom_triton_backend.py` (a **forbidden-to-edit** file), which forces
`--enforce-eager` (no CUDA graphs) and the conservative scheduling. model/dtype/TP/
max_len/util/chunked(8192)/block_size/kv_dtype/client/streaming/concurrency all match.

### V3 hit evidence during the REAL 100K request (Checkpoint F2)
Added low-overhead `num_seqs`-split counters (no in-timing sync/copy/logging). Serve log:
`FIRST num_seqs=1 V3 decode HIT: num_seqs=1 ... max_seq_len=95699` → **the real 100K
single-request decode DOES dispatch to V3** (not a fallback), no runtime-disable. So the
slow service decode is NOT a dispatch bug — it is eager-mode per-layer/per-token Python+
launch overhead (V3 runs, but 64 layers × per-token eager dispatch dominates TPOT).

### One allowed config experiment: async scheduling ON
Relaunched V4 with `--no-async-scheduling` removed (async ON confirmed in log), everything
else identical. Same ~10s prefill burst then ~2.5-min near-idle window then client
`RemoteProtocolError` — **async scheduling did NOT fix the stall/TTFT.** No improvement.

### Conclusion
The V4 score is dominated by TTFT. **V4 TTFT (153.6 s) alone exceeds the 147.0 s baseline**,
and even the baseline's own TTFT is 111.4 s (76% of budget). The +42 s TTFT / +12 ms TPOT
gap is caused by eager-mode/no-CUDA-graph, which is forced by the forbidden-file
`_cudagraph_support=NEVER`. The only legal lever (async scheduling) did not help. Beating
147 s is not achievable without editing a forbidden file or a fundamentally faster
single-request 100K prefill — neither permitted. Recorded honestly as **FAIL**; the V3
decode kernel works and is genuinely hit at 100K, but decode is not the service bottleneck.
No scoring/baseline files modified; V2/V3 kernel diff = 0.

## Phase V4 CUDA-Graph Sprint — graphs ENABLED (PIECEWISE), but 100K score still FAIL

### CUDA-graph contract (Checkpoint G1, vLLM a49d37c6b)
`AttentionCGSupport` = {NEVER=0, UNIFORM_SINGLE_TOKEN_DECODE=1, UNIFORM_BATCH=2,
ALWAYS=3}; overridden via `_cudagraph_support` on the builder. flash_attn=ALWAYS(FA3),
triton=ALWAYS. Global policy = `CUDAGraphMode` {NONE/PIECEWISE/FULL/FULL_AND_PIECEWISE}.
**Key fact: PIECEWISE captures the NON-attention layers into graphs and runs attention
eagerly at split points; it works even with backend `_cudagraph_support=NEVER`.** So the
only reason V4 had no graphs was `--enforce-eager` forcing `cudagraph_mode=NONE`.

### Minimal enablement (NO code change, NO forbidden-file edit, NO V3 graph-safety)
Launched the existing CUSTOM backend WITHOUT `--enforce-eager` (torch.compile ON,
`cudagraph_mode=FULL_AND_PIECEWISE`). vLLM auto-downgraded to **PIECEWISE** for CUSTOM
(`compilation.py:1423`: "not supported with CustomTritonBackend ... setting
cudagraph_mode=PIECEWISE"). **51 PIECEWISE graphs captured successfully**; startup
complete; 42 smoke PASS; V3 `FIRST num_seqs=1 decode HIT` confirmed (attention still runs
eagerly under PIECEWISE); no runtime-disable; no stale output.

### Results (Checkpoint G4/G5), physical GPU 3
| config | 8K TPOT | 100K TTFT | 100K TPOT | 100K score |
|---|---:|---:|---:|---:|
| baseline flash_attn | — | 111.4s | 35.5ms | 147.0s |
| V4 eager (prior) | — | 153.6s | 47.4ms | 201.0s |
| **V4 PIECEWISE graph** | **36.9ms** (was 47.4) | **153.55s** | **45.78ms** | **199.33s** |
- PIECEWISE graphs cut **8K** TPOT 47.4→36.9ms (near baseline 35.5) — graph capture is
  real and helps the short-context per-layer overhead.
- BUT **100K TTFT stayed 153.5s** (+42s over baseline) and 100K TPOT only 47.4→45.8ms.
  Score 199.3s > 147.0 → **0.7375x, FAIL**. 3/3 runs consistent (153.4/153.7/153.6s).

### Why graphs didn't fix the 100K score (Checkpoint G5 analysis)
Serve-log timeline of each 100K request: ~10s prefill burst (FA hybrid, 9569 tok/s) then
a **~2.5-min window at `generation throughput ~2.4 tok/s`, `Running: 1 req`, KV 15.8%**.
That stall = the 153s TTFT, and it is in the **eager attention path during the long
single-request 100K processing**, which PIECEWISE leaves OUTSIDE the graph. PIECEWISE only
graph-captures the non-attention layers (helps 8K where those dominate); at 100K the cost
is the eager per-token/per-chunk attention over 95k KV, unaffected by PIECEWISE. Putting
attention INSIDE a FULL-decode graph would need `_cudagraph_support>=UNIFORM_SINGLE_TOKEN_
DECODE` AND a fully graph-safe V3 (static addresses) — and even then TTFT (153s) alone
exceeds the 147s baseline, so it cannot win.

### Verdict: PERF_FAIL
CUDA graphs are correctly enabled (PIECEWISE, 51 graphs, validated: capture/replay OK,
42 PASS, V3 hits, no stale/disable) — P0 and P1 achieved. But the official 100K score
199.3s > 147.0 (P2 not met). The gap is the single-request 100K eager-attention TTFT, not
graphable. No fabricated win. V2/V3 kernel diff=0; perf_test/baseline/scoring untouched.
Enablement is a pure launch-config change (drop `--enforce-eager`); recorded in
run_final_v4_evaluation.sh for reproducibility.
