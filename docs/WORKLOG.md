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
