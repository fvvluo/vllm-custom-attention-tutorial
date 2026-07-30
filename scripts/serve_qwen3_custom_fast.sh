#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_custom_fast.sh —— 自包含的 CUSTOM 后端启动脚本（本文件由本次优化新增，
# 不改动教程原有的 serve_qwen3_custom.sh / serve_qwen3_flashattn.sh）。
#
# 解决的问题：教程原 serve_qwen3_custom.sh 不支持 HF_OVERRIDES，用户用
#   HF_OVERRIDES=... MAX_LEN=102400 ... bash serve_qwen3_custom.sh
# 跑 100k 会报 “max_model_len(102400) > derived max_model_len(40960)”（YaRN 没生效）。
# 本脚本复刻 CUSTOM 启动命令并补上 --hf-overrides（YaRN 长上下文）+ 可选优化 PROFILE（见下）。
#
# 与教程 serve_qwen3_flashattn.sh 用法一致（都支持 HF_OVERRIDES），只是 backend=CUSTOM。
# 用法（复刻用户命令，只把 flashattn 换成本脚本）：
#   HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
#   MAX_LEN=102400 GPU_MEM_UTIL=0.94 GPU=4 PORT=8004 \
#   bash scripts/serve_qwen3_custom_fast.sh
#
# 可选环境变量（本次优化）：
#   PROFILE=balanced|fp8|sparse   decode/prefill 优化组合（默认 balanced）：
#     balanced : bf16 KV + split-KV decode（无损、decode TPOT 最优；98k decode ~10x vs 无 split-KV）
#     fp8      : fp8 KV（prefill 1.7x、显存减半、近无损；decode TPOT 与 balanced 基本持平）
#     sparse   : bf16 KV + 稀疏 decode（只读 25% 块，近似；仅在极长上下文+可接受近似时用）
#   KV_CACHE_DTYPE=fp8|auto   覆盖 PROFILE 的 KV dtype 选择
#   CUSTOM_DEC_WARPS/CUSTOM_DEC_STAGES  decode kernel launch 调参（默认 4/2，实测甜点）
#   CUSTOM_PF_WINDOW=4096     滑窗 prefill（O(seq^2)->O(seq*W)，100k TTFT<50s；近似），默认 0=关
#   CUSTOM_PF_SINK=256        滑窗保留的开头 sink token 数
#   CUSTOM_SPARSE=1           手动开 decode 稀疏（PROFILE=sparse 会自动设）
# =============================================================================
set -euo pipefail

VLLM_SRC="${VLLM_SRC:-/dockerdata/landojiang/vllm_src}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"
HF_OVERRIDES="${HF_OVERRIDES:-}"

# ---- 集成优化 profile（本次）：把之前所有 decode/prefill 优化收敛成 3 个组合 ----
# 实测结论（scripts/bench_decode_tpot.py，Qwen3-32B 维度，H20）：
#   - split-KV 是 decode TPOT 的**决定性**优化（98k: 142.6→14.0 ms TPOT_attn，10.2x），**恒开**。
#   - fp8 KV 对 decode TPOT 基本中性（98k 14.3 vs 14.0 ms），主要收益在 prefill/显存。
#   - 稀疏 decode 因逐步打分+topk 固定开销，在本 kernel 下**比 dense split-KV 慢**（98k 41.4 vs 14.0），
#     仅在可接受近似且需省带宽时用。故默认 balanced（无损、最快）。
PROFILE="${PROFILE:-balanced}"
case "${PROFILE}" in
    balanced) KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}" ;;
    fp8)      KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"  ;;
    sparse)   KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"; export CUSTOM_SPARSE="${CUSTOM_SPARSE:-1}" ;;
    *) echo "未知 PROFILE=${PROFILE}（可选 balanced|fp8|sparse）" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${HF_OVERRIDES}" ]]; then
    EXTRA_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
    EXTRA_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
# WEIGHT_FP8=1：把 MLP/Linear 权重在加载时**在线量化成 fp8**（--quantization fp8）。
# MLP GEMM 是 TTFT/TPOT 里最大的非-attention项(100k TTFT 里 attention 只~4.5s、MLP~43s)；
# H20 fp8 tensor core 吞吐~bf16 2x，权重量化直接加速 MLP。近无损(per-tensor scale)，
# 但启动多一步量化(慢十几秒)、精度需 smoke 验。与 KV_CACHE_DTYPE=fp8(KV量化)正交可叠加。
WEIGHT_FP8="${WEIGHT_FP8:-0}"
if [[ "${WEIGHT_FP8}" == "1" ]]; then
    EXTRA_ARGS+=(--quantization fp8)
    echo "[serve] WEIGHT FP8 = ON (--quantization fp8, MLP/Linear 权重在线量化)" >&2
fi
# CHUNKED_PREFILL=N：开分块 prefill 并设 max_num_batched_tokens=N（把长 prompt 切成 N-token
# 的块跨调度步处理）。降低 prefill 峰值激活显存、改善并发(prefill 与其他请求 decode 交织)；
# 对单请求 TTFT 不减总算力(仍 O(seq^2))但可能改善调度重叠。默认 0=关(整段 prefill)。
CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"
if [[ "${CHUNKED_PREFILL}" != "0" ]]; then
    EXTRA_ARGS+=(--enable-chunked-prefill --max-num-batched-tokens "${CHUNKED_PREFILL}")
    echo "[serve] CHUNKED PREFILL = ON (max_num_batched_tokens=${CHUNKED_PREFILL})" >&2
fi

echo "[serve] PROFILE=${PROFILE}  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}  CUSTOM_SPARSE=${CUSTOM_SPARSE:-0}" >&2
echo "[serve] decode kernel: warps=${CUSTOM_DEC_WARPS:-4} stages=${CUSTOM_DEC_STAGES:-2} (split-KV 恒开)" >&2

# DECODE_GRAPH=1：对纯 decode 批开 CUDA graph（FULL_DECODE_ONLY），消掉 64 层逐算子 eager
# launch 开销——decode 的每步非attention时间主要是 launch 延迟而非算力，graph 是破 100tok/s
# 的关键杠杆。后端已声明 UNIFORM_SINGLE_TOKEN_DECODE + build_for_cudagraph_capture。
# 默认 0（--enforce-eager，最稳）。开 graph 时去掉 enforce-eager + async-scheduling。
DECODE_GRAPH="${DECODE_GRAPH:-0}"
MODE_ARGS=()
if [[ "${DECODE_GRAPH}" == "1" ]]; then
    MODE_ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
    echo "[serve] DECODE CUDA graph = ON (FULL_DECODE_ONLY, 去 --enforce-eager)" >&2
else
    MODE_ARGS+=(--enforce-eager --no-async-scheduling)
fi

# SPEC_NGRAM=K：开 n-gram（prompt-lookup）投机解码——draft-free，一次前向验证 K 个候选，
# 高接受率下把 decode 步数降到 ~1/(接受token数)，是突破 100 tok/s 的杠杆。验证阶段 q_len=K+1
# 走本后端 query-tiled prefill kernel（bottom-right causal 正确）。默认 0=关。
# prompt-lookup 对"输入里有可复用 n-gram"的场景（长文档问答/代码/重复模式）接受率高。
SPEC_NGRAM="${SPEC_NGRAM:-0}"
if [[ "${SPEC_NGRAM}" != "0" ]]; then
    PLMAX="${SPEC_PROMPT_LOOKUP_MAX:-4}"
    PLMIN="${SPEC_PROMPT_LOOKUP_MIN:-2}"
    MODE_ARGS+=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${SPEC_NGRAM},\"prompt_lookup_max\":${PLMAX},\"prompt_lookup_min\":${PLMIN}}")
    echo "[serve] SPEC ngram = ON (K=${SPEC_NGRAM}, prompt_lookup ${PLMIN}-${PLMAX})" >&2
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --attention-backend CUSTOM \
    "${MODE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
