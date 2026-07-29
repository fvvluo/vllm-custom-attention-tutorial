#!/usr/bin/env bash
# =============================================================================
# serve_qwen3_wzc_humaneval.sh —— HumanEval 专用：短上下文 + pad-到-128 纯 prefill
#
# HumanEval 的 prompt 很短（函数签名+docstring，几百 token），生成 <=1024 token。
# 所以每条请求 = 一次**纯 prefill**（q_len==seq_len，短）+ 若干 decode。特化如下：
#   - CHUNKED=0（--no-enable-chunked-prefill）：整条短 prompt 作为一次 pure-prefill，
#     命中适配器的 pad-到-128 路径 -> 直接调**原始** prefill kernel _wzc_attn_sparse.run
#     （见 wzc_sparse_attention.py: kernel_ok 分支）。无需矩形 causal / chunk。
#   - WZC_SPARSE_TAU=1.0：**无损**（选全部 causal 段，逐位等于 dense）。HumanEval 是
#     质量闸门（真跑代码），用无损最稳；短序列下稀疏也省不了多少，无损几乎零代价。
#   - decode -> 原始 paged decode kernel _wzc_paged_attn_decode.PagedKVDecoder.decode。
#   - MAX_LEN=8192 足够（HumanEval prompt + 1024 生成）；无需 YaRN。
#   - PIECEWISE=1（默认）：graph 化模型非 attention 部分，decode 更快。
#
# 用法：
#   GPU=0 PORT=8000 bash scripts/serve_qwen3_wzc_humaneval.sh
# 然后（另一终端）：
#   PYTHONPATH=../vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
#   PYTHONPATH=../vllm_src python scripts/humaneval_generate.py --port 8000 \
#       --model qwen3-32b --concurrency 4 --output logs/humaneval_wzc.jsonl
#   python scripts/humaneval_evaluate.py --samples logs/humaneval_wzc.jsonl \
#       --timeout 10 --workers 8 --report logs/humaneval_wzc_report.jsonl
# =============================================================================
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HumanEval 特化默认值（可被外部环境变量覆盖）。
export WZC_SPARSE_TAU="${WZC_SPARSE_TAU:-1.0}"     # 无损
export CHUNKED="${CHUNKED:-0}"                       # 纯 prefill（pad 到 128）
export PIECEWISE="${PIECEWISE:-1}"                   # decode 提速
export MAX_LEN="${MAX_LEN:-8192}"                    # 短上下文足够
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
# 不设 HF_OVERRIDES（无需 YaRN）。

exec bash "${_HERE}/serve_qwen3_wzc_sparse.sh"
