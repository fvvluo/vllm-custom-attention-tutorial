# HANDOFF — vLLM Custom Attention Tutorial (Liu Xiaochen)

## Owner / paths
- Tutorial repo: `/dockerdata/liuxiaochen/vllm_tutorial` (this repo, clone of
  `https://github.com/fvvluo/vllm-custom-attention-tutorial.git`, initial commit `002b805`)
- vLLM source tree: `/dockerdata/liuxiaochen/vllm_src` (detached HEAD `a49d37c6b`)
- Logs: `/dockerdata/liuxiaochen/vllm_tutorial/logs/` (git-ignored)
- Python: `/usr/bin/python` (3.12.13) — NOT conda.

## Verified environment (matches tutorial spec exactly)
| item | value |
|---|---|
| Python | 3.12.13 (`/usr/bin/python`) |
| torch | 2.13.0+cu130 |
| CUDA (torch) | 13.0 |
| nvcc | 13.0.88 |
| GPUs | 8× NVIDIA H20, CC 9.0, ~95 GiB each |
| installed vLLM wheel | 0.18.1rc1.dev3933+ga49d37c6b (commit a49d37c6b) |
| vLLM source HEAD | a49d37c6b |
| model | `/dockerdata/models/Qwen3-32B` (62 GB, 17 safetensors shards) |

## Source-tree setup mechanism
`scripts/setup_vllm_source_liuxiaochen.sh` clones vLLM, checks out `a49d37c6b`,
and symlinks the 16 compiled `.so` (+ `_version.py`) from the installed wheel
into the source tree. `PYTHONPATH=$VLLM_SRC` then makes `vllm.__file__` resolve
to the source tree while reusing the wheel's CUDA extensions (no recompile).
All 16 symlinks valid, 0 broken. `import vllm._C_stable_libtorch` OK.

## Plugin
`pip install -e .` registers the general-plugin entry point
`custom_triton -> custom_backend.plugin:register`, which maps
`AttentionBackendEnum.CUSTOM` to `custom_backend.custom_triton_backend.CustomTritonBackend`.
`custom_backend.__file__` = tutorial dir.

## GPU / port usage rules (personal)
- Always pass explicit `GPU=<idx>` and `PORT=<port>` (serve scripts require them).
- Never default to GPU 0 / port 8000. Pick a GPU with ~0% util and no other
  user's compute process; Qwen3-32B needs ~62 GB weights + KV.
- Foreground serve scripts; stop with SIGINT; verify port + GPU released.
- Ports used so far: 8401 (flashattn baseline), 8402 (custom). Both released.

## Baseline results (Phase V1, all PASS)
- FlashAttention serve (GPU4/8401, MAX_LEN=8192): backend=flash_attn,
  Application startup complete, smoke_test → 42, `[smoke] PASS`.
- CUSTOM serve (GPU4/8402): "Using AttentionBackendEnum.CUSTOM backend",
  smoke_test → 42, `[smoke] PASS`.
- Paged-attn correctness (`tests/test_paged_attn_correctness.py`, GPU4, bf16 rtol=atol=2e-2):
  prefill max_abs=7.81e-3 PASS, decode 3.89e-3 PASS, mixed 7.75e-3 PASS → ALL PASS.

## Logs
- Setup: `/tmp/liuxiaochen_vllm_setup.log`
- FlashAttn serve: `logs/serve_flashattn_gpu4_8401.log`
- CUSTOM serve: `logs/serve_custom_gpu4_8402.log`
- Paged correctness: `/tmp/liuxiaochen_paged_attn_baseline.log`
- Tutorial initial commit: `/tmp/liuxiaochen_vllm_tutorial_initial_commit.txt`

## B7 status
No B7 (old attention-test warp-MMA decode) code has been ported into this
project. The old repo `/dockerdata/attention-test-liuxiaochen` is untouched.
See `docs/DECISIONS.md` for what is reusable vs not.

## Next phase (planned, NOT started)
Study `paged_attention_triton` interface; later replace the teaching Triton
kernel with an optimized custom kernel adapted to vLLM's packed-query + paged
KV-cache layout. No performance optimization / 100k perf test this phase.

## V2 / V3 status (standalone paged-decode prototype — NOT wired to CUSTOM backend)
Directory: `custom_backend/liuxiaochen_paged_decode/`. See its `README.md` for full
detail. Both are decode-only, fixed target (BF16, D=128, Hq=64, Hkv=8, block_size=16,
SM90), directly reading the paged KV-cache via `block_table` (no gather/copy/fallback).

- **V2** (frozen baseline, committed `954e8b0`): synchronous per-element paged K/V
  load. warp-MMA m16n8k16 QK/PV + online softmax + Split-KV + LSE combine. 128K
  split=256 = 3.221 ms e2e (24.3x tutorial-Triton), ~167 GB/s. Bottleneck = sync load.
- **V3** (this round, UNCOMMITTED): V2 with the ONE variable changed — sync load →
  **cp.async 2-stage (2K+2V) double-buffered** paged load. Prefetch tile t+1 while
  computing tile t. **Bit-identical to V2** in all correctness cases. 128K split=256
  = **0.245 ms e2e (13.15x over V2, 319x over Triton)**, **2.29 TB/s (57% DRAM peak)**.
  ncu: HMMA unchanged, long-scoreboard stall 20.9→0.86, no spill. block_table
  shuffle penalty <1%.
- Files: `paged_decode_stage1_v{2,3}.py`, `runner_v{2,3}.py`, `verify_paged_decode_v{2,3}.py`,
  `bench_paged_decode_v{2,3}.py`, `paged_decode_combine_v2.py` (shared), `ncu_stage1_probe.py`.
- Logs: `logs/v3_verify_all.log`, `v3_verify_big.log`, `v3_bench_128k_split_sweep.log`,
  `v3_bench_seqlens.log`, `v3_bench_blocktable.log`, `v3_ncu_v2.log`, `v3_ncu_v3_full.log`.
- Pre-V3 archive: `/tmp/liuxiaochen_before_v3_cpasync.tar.gz`
  (sha256 `6a7d3ace43071045db15fc93da68eb0e3d34d3b5732cc564c7698529f7e5b9bd`).

### V3 evidence boundary (log-backed vs author-asserted vs pending)
- **A. Log-backed** (`logs/v3_*.log`): V3==V2 bit-identical; correctness matrix PASS
  (2e-2 + strict 5e-3); benchmark `EXIT=0`; 128K split=256 e2e≈0.245 ms; Nsight metric
  values; identity/shuffle penalties; no register spill.
- **B. Author-asserted, NOT in any log** (do NOT treat as verified): the physical GPU
  ("GPU 1/6" — logs only show `device: cuda:0`, no physical index / no
  `CUDA_VISIBLE_DEVICES`); the exact full command lines; the real **ncu shell exit
  codes** (ncu logs have no `EXIT=` sentinel).
- **C. NOT executed (no artifacts):** V3.1 two-GPU independent-process retest; V3.1
  Part VI length scaling; pointer rebinding / stale workspace audit; Nsight Systems
  timeline audit. None of B or C may be reported as PASS.

> V3.1 measurement-integrity audit remains pending and should be
> performed in the dedicated per-user container or another clean GPU
> environment. This does not invalidate the existing V3 correctness,
> benchmark, or Nsight evidence.

- Next (out of scope this round): deeper cp.async pipeline / PDL / wire into CUSTOM backend.
