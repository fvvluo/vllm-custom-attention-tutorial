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
