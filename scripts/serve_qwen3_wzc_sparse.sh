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
# 阶段说明（见 BENCHMARK_OPTIMIZATION_DESIGN.md）：
#   - 阶段 0（解阻塞）：PIECEWISE=0 → 纯 eager（--enforce-eager），只验证稀疏 kernel 触发。
#   - 阶段 A1（压 TPOT）：PIECEWISE=1（默认）→ VLLM_COMPILE + cudagraph_mode=PIECEWISE，
#     把 attention 之外的模型部分 graph 化（decode TPOT 主瓶颈），attention 段间仍 eager。
#     后端 _cudagraph_support 保持 NEVER，无需改 kernel。
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

# ---- 阶段 A1：PIECEWISE CUDA graph（压 TPOT，不改 kernel）----
# PIECEWISE=1（默认）：去掉 --enforce-eager，用 VLLM_COMPILE + cudagraph_mode=PIECEWISE。
#   把模型除 attention 外的部分（QKV/MLP/norm/router，上百次 kernel launch）graph 化，
#   attention 段间仍跑 eager —— 正好捕获 decode TPOT 的真正瓶颈（逐 kernel 启动开销），
#   而 wzc 逐请求 python 循环 attention 原样保留（在 graph 之外跑）。后端 _cudagraph_support
#   保持 NEVER：vLLM 见到 NEVER 会自动把 FULL 降级到 PIECEWISE（compilation.py 实测），
#   这里显式写死 PIECEWISE 更稳、启动日志更清晰。
# PIECEWISE=0：回退老的纯 eager 路径（--enforce-eager），用于 A/B 对比 TPOT。
PIECEWISE="${PIECEWISE:-1}"
if [[ "${PIECEWISE}" == "1" ]]; then
    # 显式 VLLM_COMPILE + PIECEWISE；不加 --enforce-eager。
    EXTRA_ARGS+=(-cc '{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE"}')
    # piecewise 下异步调度可保持默认开启（eager 后端才需要 --no-async-scheduling）；
    # 若观察到请求卡住再手动加 EXTRA_SERVE_ARGS="--no-async-scheduling"。
    EAGER_ARGS=()
else
    # 纯 eager 回退（阶段 0 用的路径）。
    EAGER_ARGS=(--enforce-eager --no-async-scheduling)
fi
# 允许外部追加任意 serve 参数（如 --no-async-scheduling）。
read -r -a _EXTRA_SERVE <<< "${EXTRA_SERVE_ARGS:-}"

# ---- chunked prefill 开关（阶段 B 用）----
# CHUNKED=0（默认）：--no-enable-chunked-prefill —— 整段 prompt 一次 pure-prefill 触发
#   square 稀疏 kernel。单卡 H20 上限 ~32k（超了激活+KV 抢显存 OOM，见设计文档 §5 轨-0）。
# CHUNKED=1：用 vLLM 默认 chunked prefill —— 100k prompt 切成 chunk（激活受限，显存够）。
#   chunk0 是 pure-prefill（q_len==seq_len，走稀疏 kernel）；后续 chunk 有 context
#   （q_len<seq_len），当前适配器走精确 torch 回退（正确但慢）——这是阶段 B 要用矩形
#   causal 稀疏 kernel 替换的部分。用它建立“chunked ON 正确性/性能基线”。
CHUNKED="${CHUNKED:-0}"
CHUNK_ARGS=()
if [[ "${CHUNKED}" == "1" ]]; then
    : # 不加 --no-enable-chunked-prefill，用 vLLM 默认（开启）。可选调 chunk 大小：
    [[ -n "${MAX_NUM_BATCHED_TOKENS_OVERRIDE:-}" ]] && \
        CHUNK_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS_OVERRIDE}")
else
    CHUNK_ARGS+=(--no-enable-chunked-prefill \
                 --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --attention-backend CUSTOM \
    "${CHUNK_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${_EXTRA_SERVE[@]}"
