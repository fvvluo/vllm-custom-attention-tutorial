#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_flashattn_100k.sh —— serve_qwen3_flashattn.sh 的“数值写死”克隆版
#
# 本文件是 serve_qwen3_flashattn.sh 的克隆（原脚本保持不动）。区别：去掉了所有
# 环境变量接口（${VAR:-default}），把 100k 性能测试用的取值直接写死，等价于：
#   MAX_LEN=102400 GPU_MEM_UTIL=0.94 GPU=6 PORT=8004 bash scripts/serve_qwen3_flashattn.sh
# 用法（零参数）：
#   bash scripts/serve_qwen3_flashattn_100k.sh
# 然后另一终端：
#   python scripts/perf_test.py --port 8004 --input-len 100000 --output-len 64
#
# 后端仍是 flash_attn（README 基线，~111s TTFT）。若要 wzc 加速版一键脚本见
# serve_qwen3_wzc_fast.sh。
# 前置：先跑过 scripts/setup_vllm_source.sh。
# =============================================================================
set -euo pipefail

# ---- 写死的取值（原脚本这些都是 ${VAR:-...} 可变；此处固定）----
VLLM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vllm_src"
MODEL="/dockerdata/models/Qwen3-32B"
SERVED_NAME="qwen3-32b"
PORT="8004"
GPU="6"
GPU_MEM_UTIL="0.94"
MAX_LEN="102400"
# YaRN rope 扩展到 128k+（100k 输入必需；原生上下文仅 ~40960）。
HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"

# 该 commit 用 --attention-backend 选后端；flash_attn 是默认，这里显式写出以作教学。
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --attention-backend flash_attn \
    --hf-overrides "${HF_OVERRIDES}"
