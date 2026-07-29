#!/usr/bin/env bash
# fp8 online weight-quant + fp8 KV serve (task2/3 best config) — linqihao workspace
# + optional CHUNK knob = --max-num-batched-tokens (chunked-prefill chunk size).
#   Larger chunk => fewer prefill scheduler round-trips + better GEMM util => lower TTFT.
#   PURE SCHEDULING lever: numeric path unchanged => HumanEval pass@1 bit-identical.
set -euo pipefail
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONPATH=/dockerdata/linqihao/vllm_src
PORT="${PORT:-8000}"
QUANT="${QUANT:-fp8}"
KVDTYPE="${KVDTYPE:-fp8}"
MEM="${MEM:-0.90}"
MAXLEN="${MAXLEN:-131072}"
CHUNK="${CHUNK:-}"   # empty => vLLM default (2048); set e.g. 8192/16384/131072

EXTRA=()
if [[ "${KVDTYPE}" != "auto" ]]; then
  EXTRA+=(--kv-cache-dtype "${KVDTYPE}")
fi
if [[ -n "${QUANT}" && "${QUANT}" != "none" ]]; then
  EXTRA+=(--quantization "${QUANT}")
fi
if [[ -n "${CHUNK}" ]]; then
  EXTRA+=(--max-num-batched-tokens "${CHUNK}")
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model /dockerdata/models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --port "${PORT}" \
  --gpu-memory-utilization "${MEM}" \
  --max-model-len "${MAXLEN}" \
  --attention-backend flash_attn \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
  "${EXTRA[@]}"
