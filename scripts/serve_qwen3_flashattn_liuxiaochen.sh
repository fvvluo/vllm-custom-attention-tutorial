#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_flashattn_liuxiaochen.sh
# Personalized copy of serve_qwen3_flashattn.sh (author: landojiang).
# ONLY changes vs original: personal path defaults + mandatory GPU/PORT +
# pre-launch safety checks + personal PYTHONPATH + logs dir. Serving semantics
# (flash_attn backend, single-card) unchanged. Foreground only; no nohup/kill.
# =============================================================================
set -euo pipefail

TUTORIAL_DIR="${TUTORIAL_DIR:-/dockerdata/liuxiaochen/vllm_tutorial}"
VLLM_SRC="${VLLM_SRC:-/dockerdata/liuxiaochen/vllm_src}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
GPU="${GPU:?GPU must be explicitly specified (physical index)}"
PORT="${PORT:?PORT must be explicitly specified}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
LOG_DIR="${LOG_DIR:-${TUTORIAL_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# ---- safety checks ----
NGPU="$(/usr/bin/python -c 'import torch;print(torch.cuda.device_count())')"
if ! [[ "${GPU}" =~ ^[0-9]+$ ]] || [ "${GPU}" -ge "${NGPU}" ]; then
    echo "ERROR: GPU=${GPU} invalid (device count=${NGPU})" >&2; exit 2
fi
echo "==> target GPU ${GPU} current processes:"
nvidia-smi -i "${GPU}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
echo "==> target GPU ${GPU} util/mem:"
nvidia-smi -i "${GPU}" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
    echo "ERROR: PORT ${PORT} already in use" >&2; exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${TUTORIAL_DIR}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${HF_OVERRIDES}" ]]; then
    EXTRA_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

echo "==> launching FLASH_ATTN server: GPU=${GPU} PORT=${PORT} MAX_LEN=${MAX_LEN}"
echo "    PYTHONPATH=${PYTHONPATH}"
set -x
exec /usr/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --attention-backend flash_attn \
    "${EXTRA_ARGS[@]}"
