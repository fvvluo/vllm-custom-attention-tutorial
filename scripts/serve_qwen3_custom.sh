#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_custom.sh —— 单卡启动 Qwen3-32B，使用自定义 CUSTOM attn backend
#
# 前置：
#   1) 已跑过 scripts/setup_vllm_source.sh
#   2) 已在教程目录 pip install -e . （注册 vllm.general_plugins 入口）
# 用法：
#   bash scripts/serve_qwen3_custom.sh
# =============================================================================
set -euo pipefail

VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vllm_src}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"
# CUSTOM 后端通过 pip 安装的 general_plugin 自动注册（见 pyproject.toml）。
# 教学示例后端不参与 CUDA graph，用 --enforce-eager 关闭 torch.compile / graph 捕获，
# 路径最简单、最好调试。
#
# 为什么加 --no-async-scheduling：
#   vLLM 默认开启异步调度（在 GPU 跑第 N 步的同时，CPU 提前调度第 N+1 步）。
#   这个重叠对“不参与 CUDA graph 的简易 eager 后端”不友好，实测会导致请求卡住
#   （GPU 占用 0%、请求不被调度）。关掉异步调度后，请求几秒内即可正确返回。
#   （前缀缓存 prefix caching 可保持默认开启，不是卡住的原因。）
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --enforce-eager \
    --no-async-scheduling \
    --attention-backend CUSTOM
