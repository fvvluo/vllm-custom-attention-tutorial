#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_flashattn.sh —— 单卡启动 Qwen3-32B（默认 flash attention 后端）
#
# 前置：先跑过 scripts/setup_vllm_source.sh。
# 用法：
#   bash scripts/serve_qwen3_flashattn.sh
# 服务起在 http://127.0.0.1:8000 ，another shell 里跑 smoke_test.py 验证。
# =============================================================================
set -euo pipefail

VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vllm_src}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"                 # 用哪张卡
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"      # 单卡演示用，够跑 human eval / 冒烟测试
# 可选：HF config 覆盖（JSON）。用于长上下文性能测试时开 YaRN rope 扩展到 128k。
# 该检查点 config 里 max_position_embeddings=40960（原生 ~40k），要跑 100k 输入需设：
#   HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}'
HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${HF_OVERRIDES}" ]]; then
    EXTRA_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

# 该 commit 用 --attention-backend 选后端；flash_attn 是默认，这里显式写出以作教学。
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --attention-backend flash_attn \
    "${EXTRA_ARGS[@]}"
