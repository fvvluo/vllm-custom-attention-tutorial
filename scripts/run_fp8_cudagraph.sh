#!/usr/bin/env bash
# 实验: 用 PIECEWISE CUDA graph (合规,不改只读文件,只用命令行)
# attention走eager(CUSTOM=NEVER), 其余63层图化 -> 降decode Python分发开销
# 用法: bash run_fp8_cudagraph.sh <GPU> <PORT> [PREFILL_WINDOW]
# v18最优默认: PREFILL_WINDOW=128, DECODE_WINDOW=1024, MAX_BATCHED_TOKENS=32768
#   → 稳定E2E~38s/3.85×, 最好36.5s/4.03×, HumanEval 89.02%, correctness ALL PASS
set -uo pipefail
GPU="${1:-0}"; PORT="${2:-8000}"; WIN="${3:-128}"
LOG="/dockerdata/quanbofeng/task3/logs/serve_cudagraph_gpu${GPU}_$(date +%H%M%S).log"
echo "LOGFILE=$LOG"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/dockerdata/landojiang/vllm_src
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
export PREFILL_WINDOW="$WIN"
export VLLM_USE_DEEP_GEMM=1
# decode窗口稀疏默认1024(env可覆盖); 未设时用v18最优值
export DECODE_WINDOW="${DECODE_WINDOW:-1024}"

# 可调参（env 覆盖）：chunked-prefill 的 chunk 大小、显存利用率、max-model-len
MAX_BATCHED="${MAX_BATCHED_TOKENS:-32768}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAXLEN="${MAXLEN:-102400}"

# 关键: 不用 --enforce-eager; mode=3(VLLM_COMPILE)使attention被piecewise切分,
# 触发源码里 "setting cudagraph_mode=PIECEWISE because attention is compiled piecewise"
exec /usr/bin/python -m vllm.entrypoints.openai.api_server \
    --model /dockerdata/models/Qwen3-32B-FP8 \
    --served-model-name qwen3-32b --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAX_BATCHED" \
    --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
    --no-async-scheduling \
    --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
    --attention-backend CUSTOM \
    > "$LOG" 2>&1

# 关键: 不用 --enforce-eager; mode=3(VLLM_COMPILE)使attention被piecewise切分,
# 触发源码里 "setting cudagraph_mode=PIECEWISE because attention is compiled piecewise"
exec /usr/bin/python -m vllm.entrypoints.openai.api_server \
    --model /dockerdata/models/Qwen3-32B-FP8 \
    --served-model-name qwen3-32b --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAX_BATCHED" \
    --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
    --no-async-scheduling \
    --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}' \
    --attention-backend CUSTOM \
    > "$LOG" 2>&1
