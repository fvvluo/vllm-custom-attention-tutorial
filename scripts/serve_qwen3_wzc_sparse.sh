#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_wzc_sparse.sh —— 单卡启动 Qwen3-32B，用 CUSTOM 后端 + wzc 稀疏 kernel
#
# 与教学版 serve_qwen3_custom.sh 的区别（本脚本是 wzc 专用，不改教学脚本）：
#   1) WZC_SPARSE_BACKEND=1 —— 让 CustomTritonImpl.forward 走 wzc 的 block-top-k
#      稀疏 prefill kernel + paged decode kernel（见 custom_backend/wzc_sparse_attention.py）。
#   2) --no-enable-chunked-prefill + --max-num-batched-tokens=MAX_LEN —— vLLM 默认
#      开 chunked prefill（max_num_batched_tokens=2048），会把 100k prompt 切成 2048
#      的 chunk，使 q_len<seq_len，稀疏 prefill kernel（square-causal，要 q_len==seq_len）
#      永远落到 torch 慢路径。关掉分块 + 把单批 token 上限调到 >= max_model_len，
#      让整段 100k prompt 作为“一次 pure-prefill”进 forward，稀疏 kernel 才会真正触发。
#   3) HF_OVERRIDES 支持 —— 100k 输入需 YaRN 把上下文从原生 ~40k 扩到 128k+（同
#      serve_qwen3_flashattn.sh）。跑短上下文冒烟测试时可不设。
#
# 阶段说明（见 BENCHMARK_OPTIMIZATION_DESIGN.md）：本脚本是“阶段 0（解阻塞）”用的。
#   仍保留 --enforce-eager（decode 走 eager，TPOT 项此时不看）；CUDA graph 是阶段 A 的事。
#
# 前置：
#   1) 已跑过 scripts/setup_vllm_source.sh
#   2) 已在教程目录 pip install -e . （注册 vllm.general_plugins 入口）
#
# 用法：
#   # 短上下文冒烟（不需要 YaRN）：
#   GPU=2 PORT=8002 bash scripts/serve_qwen3_wzc_sparse.sh
#   # 100k 性能测试（开 YaRN + 调大 MAX_LEN）：
#   HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
#     MAX_LEN=102400 GPU_MEM_UTIL=0.94 GPU=2 PORT=8002 bash scripts/serve_qwen3_wzc_sparse.sh
# =============================================================================
set -euo pipefail

VLLM_SRC="${VLLM_SRC:-/dockerdata/landojiang/vllm_src}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"
# 单批 token 上限：默认跟随 MAX_LEN，保证整段 prompt 能一次进 forward（不被分块）。
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_LEN}}"
# 可选：HF config 覆盖（JSON），用于 100k 长上下文时开 YaRN rope 扩展。
HF_OVERRIDES="${HF_OVERRIDES:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"
# 关键：切到 wzc 稀疏后端（见 custom_backend/custom_triton_backend.py 的分发）。
export WZC_SPARSE_BACKEND=1
# 可选：打开路由计数器，确认稀疏 kernel 真触发（日志出现 [wzc-stats] kernel_reqs=...）。
export WZC_SPARSE_STATS="${WZC_SPARSE_STATS:-1}"

EXTRA_ARGS=()
if [[ -n "${HF_OVERRIDES}" ]]; then
    EXTRA_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

# --enforce-eager / --no-async-scheduling 的原因同教学脚本（eager 后端不参与 CUDA graph）。
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-chunked-prefill \
    --enforce-eager \
    --no-async-scheduling \
    --attention-backend CUSTOM \
    "${EXTRA_ARGS[@]}"
