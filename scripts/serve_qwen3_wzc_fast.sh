#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_wzc_fast.sh —— 一键起最优 wzc 加速服务（所有超参已写死）
#
# 用法（零额外参数）：
#   GPU=<空闲卡> PORT=8004 bash scripts/serve_qwen3_wzc_fast.sh
# 然后另一终端直接：
#   python scripts/perf_test.py --port 8004 --input-len 100000 --output-len 64
#   （perf_test 默认 --warmup 1 --repeat 3，自动排除一次性编译开销，测稳态 TTFT）
#
# 这里把本轮调优确定的最优组合全部固化，无需再传任何超参：
#   - CUSTOM 后端（triton_attention.py 已委托 wzc 稀疏 prefill + paged decode）
#   - QUANT=fp8         : 在线权重量化，H20 fp8 tensor core，砍 MLP/GEMM 地板（~43s->~21s）
#   - tau=0.5           : block-top-k 稀疏（HumanEval 无损，见代码 _TAU 默认值）
#   - chunk=16384       : max_num_batched_tokens 甜点（默认 2048 每-chunk 开销过大；32768 反退化）
#   - block_size=128    : 分页块=kernel page_size
#   - PIECEWISE cudagraph: graph 化非 attention 部分，压 decode TPOT
#   - YaRN factor=4.0   : 上下文扩到 128k+，支持 100k 输入
#   - VLLM_SRC 完整树    : fp8 路径需 flashmla，用 landojiang 完整源码
# 实测（单张 H20 独占，稳态）：TTFT 111s(flash) -> 32.8s（3.4x），HumanEval pass@1 无损。
# =============================================================================
set -euo pipefail

# 完整 vLLM 源码树（fp8 的 MLA import 需要 flashmla；兄弟目录克隆不完整）。
export VLLM_SRC="${VLLM_SRC:-/dockerdata/landojiang/vllm_src}"
# 固化的最优超参（可被环境变量覆盖，但默认即最优，无需传）。
export QUANT="${QUANT:-fp8}"
export CHUNKED="${CHUNKED:-1}"
export MAX_NUM_BATCHED_TOKENS_OVERRIDE="${MAX_NUM_BATCHED_TOKENS_OVERRIDE:-16384}"
export PIECEWISE="${PIECEWISE:-1}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}"
export MAX_LEN="${MAX_LEN:-98304}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# tau 由 custom_backend/wzc_sparse_attention.py 的 _TAU 默认（0.5）决定，无需在此设。
# YaRN：100k 输入必需（原生上下文仅 ~40k）。JSON 原文，不带外层引号（父脚本会加）。
export HF_OVERRIDES="${HF_OVERRIDES:-{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":40960},\"max_position_embeddings\":163840}}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${_HERE}/serve_qwen3_wzc_sparse.sh"
