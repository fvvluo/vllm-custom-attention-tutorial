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
