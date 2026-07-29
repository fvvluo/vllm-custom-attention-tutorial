#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_wzc_sparse.sh —— 单卡启动 Qwen3-32B，CUSTOM 后端（wzc 稀疏 kernel）
#
# 契约合规（README Part 2.5）：wzc kernel 已接在 `custom_backend/triton_attention.py`
# 的 `paged_attention_triton` 里（只改这一个文件），`custom_triton_backend.py` /
# `plugin.py` / `__init__.py` / `pyproject.toml` 全部保持 origin 原样。因此本脚本
# 无需任何 WZC 环境开关——CUSTOM 后端天然就走 wzc kernel。
#
# 本脚本相对教学版 serve_qwen3_custom.sh 的额外项：
#   1) --block-size 128 —— 让 vLLM 用 128 的分页块，物理页布局直接等于 wzc paged-decode
#      kernel 的 128-page pool -> decode 走零拷贝路径（不再每步 gather+repack 整段历史，
#      100k context 下把 TPOT 从 ~260ms 降回 ~40ms）。这是纯 CLI 参数，不碰任何后端文件。
#   2) PIECEWISE=1（默认）—— VLLM_COMPILE + cudagraph_mode=PIECEWISE，graph 化 attention
#      之外的模型部分（decode TPOT 主瓶颈），attention 段间仍 eager。见设计文档阶段 A1。
#   3) CHUNKED / HF_OVERRIDES —— 见下方开关；100k 输入需 YaRN + chunked prefill。
#
# 可选诊断：WZC_SPARSE_STATS=1 让 wzc 适配器打印 [wzc-stats] 路由计数（确认稀疏 kernel 触发）。
#
# 前置：
#   1) 已跑过 scripts/setup_vllm_source.sh
#   2) 已在教程目录 pip install -e . （注册 vllm.general_plugins 入口）
#
# 用法：
#   # 短上下文冒烟（不需要 YaRN）：
#   GPU=2 PORT=8002 bash scripts/serve_qwen3_wzc_sparse.sh
#   # 100k 性能测试（开 YaRN + chunked + 调大 MAX_LEN）：
#   HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
#     CHUNKED=1 MAX_LEN=98304 GPU_MEM_UTIL=0.95 GPU=2 PORT=8002 bash scripts/serve_qwen3_wzc_sparse.sh
# =============================================================================
set -euo pipefail

# vLLM 源码：优先用 origin 约定的兄弟目录 ../../vllm_src（相对本脚本）；
# 若不存在则回退到镜像内旧路径（本机仍存在）。也可用 VLLM_SRC 覆盖。
_SIBLING_VLLM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vllm_src"
if [[ -n "${VLLM_SRC:-}" ]]; then
    :
elif [[ -d "${_SIBLING_VLLM}" ]]; then
    VLLM_SRC="${_SIBLING_VLLM}"
else
    VLLM_SRC="/dockerdata/landojiang/vllm_src"
fi
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"
# 单批 token 上限：默认跟随 MAX_LEN，保证整段 prompt 能一次进 forward（不被分块）。
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_LEN}}"
# 分页块大小：128 使 KV cache 物理页布局 == wzc paged-decode kernel 的 128-page pool。
BLOCK_SIZE="${BLOCK_SIZE:-128}"
# 可选：HF config 覆盖（JSON），用于 100k 长上下文时开 YaRN rope 扩展。
HF_OVERRIDES="${HF_OVERRIDES:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"
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

# ---- 可选 fp8 权重量化（压 MLP/QKVO GEMM 地板）----
# QUANT=fp8：在线动态量化 bf16 checkpoint 的线性层权重到 fp8，走 H20 fp8 tensor core
#   (~2×)，把 100k prefill 里 ~43s 的 MLP/GEMM 地板降到 ~21s（attention 由 wzc 稀疏另算）。
#   有损，需 HumanEval 守质量。QUANT 留空则 bf16（默认）。
QUANT="${QUANT:-}"
QUANT_ARGS=()
[[ -n "${QUANT}" ]] && QUANT_ARGS+=(--quantization "${QUANT}")

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --block-size "${BLOCK_SIZE}" \
    --attention-backend CUSTOM \
    "${QUANT_ARGS[@]}" \
    "${CHUNK_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${_EXTRA_SERVE[@]}"
