#!/usr/bin/env bash
# =============================================================================
# run_final_v4_evaluation.sh — one-shot official V4 evaluation (Liu Xiaochen)
#
# Starts the V4 CUSTOM service (V3 paged-decode dispatch ON), runs the OFFICIAL
# perf_test.py (100K input / 64 output, default args, official baseline 147s) and
# HumanEval pass@1, and cleans up ONLY the service PID it started.
#
# Usage:
#   bash scripts/run_final_v4_evaluation.sh <PHYSICAL_GPU> [PORT]
#
# Discipline: no pkill -f; never kills other users' processes; no param downgrade;
# no FP8; no auto-fallback to smaller input; official default bench args only.
# =============================================================================
set -uo pipefail

TUTORIAL_DIR="${TUTORIAL_DIR:-/dockerdata/liuxiaochen/vllm_tutorial}"
MODEL="${MODEL:-/dockerdata/models/Qwen3-32B}"
GPU_ID="${1:?usage: run_final_v4_evaluation.sh <PHYSICAL_GPU> [PORT]}"
PORT="${2:-8403}"
# Official 100K service needs near-full card AND YaRN RoPE scaling (Qwen3-32B native
# max_position_embeddings=40960; 100K input requires the official HF_OVERRIDES YaRN
# override, exactly as scripts/serve_qwen3_custom.sh documents and the flash_attn
# baseline uses). MAX_LEN 102400 covers 100K input + 64 output. util 0.95: at 0.9 the
# KV cache (22.09 GiB) is just short of the 25 GiB needed for a full 102400-token seq;
# 0.95 gives enough KV (matches the foreign 100K servers' 0.95/0.97).
MAX_LEN="${MAX_LEN:-102400}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":40960},\"max_position_embeddings\":163840}}"
LOG_DIR="${LOG_DIR:-${TUTORIAL_DIR}/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVE_LOG="${LOG_DIR}/v4_final_serve_gpu${GPU_ID}_${PORT}_${STAMP}.log"
PERF_LOG="${LOG_DIR}/v4_final_perf_gpu${GPU_ID}_${STAMP}.log"
HE_GEN_LOG="${LOG_DIR}/v4_final_humaneval_gen_${STAMP}.log"
HE_EVAL_LOG="${LOG_DIR}/v4_final_humaneval_eval_${STAMP}.log"
HE_SAMPLES="${LOG_DIR}/v4_final_humaneval_samples_${STAMP}.jsonl"
mkdir -p "${LOG_DIR}"

SERVER_PID=""
cleanup() {
    # ONLY terminate the service PID this script started.
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval] stopping own service PID ${SERVER_PID}"
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
        kill -0 "${SERVER_PID}" 2>/dev/null && kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---- 2. GPU memory precheck ----
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
echo "[eval] GPU ${GPU_ID} free=${FREE_MIB} MiB (need ~>=75000 for official 100K BF16)"
if [ "${FREE_MIB}" -lt 75000 ]; then
    echo "[eval] RESOURCE_BLOCKED: GPU ${GPU_ID} free ${FREE_MIB} MiB < 75000; not starting."
    exit 20
fi

# ---- 3. port check ----
if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
    echo "[eval] ERROR: port ${PORT} in use"; exit 2
fi

# ---- 4. no_proxy ----
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY="${no_proxy}"

# ---- 5. start V4 service (feature flag ON; low-freq debug only) ----
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="/dockerdata/liuxiaochen/vllm_src:${TUTORIAL_DIR}:${PYTHONPATH:-}"
export LIUXIAOCHEN_PAGED_DECODE_V3=1
unset LIUXIAOCHEN_PAGED_DECODE_V3_DEBUG   # keep only FIRST hit / first fallback / totals

# CUDA graphs: dropping --enforce-eager lets vLLM run torch.compile + PIECEWISE CUDA
# graphs (auto-downgraded from FULL_AND_PIECEWISE because CustomTritonBackend declares
# AttentionCGSupport.NEVER — attention runs eagerly at split points, non-attention layers
# are graph-captured). Matches the baseline's non-eager config. Set ENFORCE_EAGER=1 to
# force the old eager path. async scheduling left at vLLM default (ON, like baseline).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_ARGS=()
[ "${ENFORCE_EAGER}" = "1" ] && EAGER_ARGS+=(--enforce-eager --no-async-scheduling)

SERVE_CMD="/usr/bin/python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL} --served-model-name qwen3-32b --port ${PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} --max-model-len ${MAX_LEN} \
  ${EAGER_ARGS[*]} --attention-backend CUSTOM \
  --hf-overrides ${HF_OVERRIDES}"
echo "[eval] SERVE CMD: CUDA_VISIBLE_DEVICES=${GPU_ID} LIUXIAOCHEN_PAGED_DECODE_V3=1 ${SERVE_CMD}" | tee "${SERVE_LOG}"
/usr/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" --served-model-name qwen3-32b --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" --max-model-len "${MAX_LEN}" \
  "${EAGER_ARGS[@]}" --attention-backend CUSTOM \
  --hf-overrides "${HF_OVERRIDES}" >> "${SERVE_LOG}" 2>&1 &
SERVER_PID=$!
echo "[eval] service PID=${SERVER_PID} log=${SERVE_LOG}"

# ---- 7. wait ready (poll /v1/models; fail fast if the process dies) ----
BASE="http://127.0.0.1:${PORT}"
READY=0
for _ in $(seq 1 360); do   # up to ~30 min
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval] service process exited during startup — see ${SERVE_LOG}"; exit 21
    fi
    if curl -s "${BASE}/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    sleep 5
done
[ "${READY}" -eq 1 ] || { echo "[eval] service not ready in time"; exit 22; }
echo "[eval] service READY"

# ---- 8. smoke: 42 + >=64 tokens + 2 requests ----
echo "[eval] smoke 42..."
/usr/bin/python "${TUTORIAL_DIR}/scripts/smoke_test.py" --port "${PORT}" || { echo "[eval] smoke42 FAIL"; exit 23; }
echo "[eval] smoke 64-token generation..."
/usr/bin/python - "${PORT}" <<'PYEOF' || { echo "[eval] gen smoke FAIL"; exit 24; }
import sys, json, urllib.request
port = sys.argv[1]
def chat(msg, mx):
    body = {"model":"qwen3-32b","messages":[{"role":"user","content":msg}],
            "max_tokens":mx,"temperature":0.0,
            "chat_template_kwargs":{"enable_thinking":False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read()); return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]
c1,n1 = chat("Write a short poem about the sea in 8 lines.", 80)
c2,n2 = chat("List the first 10 prime numbers, comma separated.", 64)
print(f"[gen] req1 tokens={n1} nonempty={bool(c1.strip())}")
print(f"[gen] req2 tokens={n2} nonempty={bool(c2.strip())}")
assert n1 >= 32 and c1.strip() and c2.strip(), "generation smoke insufficient"
print("[gen] PASS")
PYEOF

# ---- 10/11. dispatch evidence ----
echo "[eval] === dispatch evidence in serve log ==="
grep -nE "FIRST V3 decode HIT|fallback \(first of kind\)|v4-dispatch\] TOTALS|runtime, DISABLING" "${SERVE_LOG}" || true
if ! grep -q "FIRST V3 decode HIT" "${SERVE_LOG}"; then
    echo "[eval] WARNING: no V3 decode HIT seen yet (will re-check after perf run)"
fi

# ---- 12. official 100K benchmark (DEFAULT args, official baseline) ----
echo "[eval] official 100K perf (input 100000 / output 64, warmup 1 / repeat 3)..."
/usr/bin/python "${TUTORIAL_DIR}/scripts/perf_test.py" \
    --port "${PORT}" --input-len 100000 --output-len 64 --warmup 1 --repeat 3 \
    2>&1 | tee "${PERF_LOG}"
PERF_STATUS=${PIPESTATUS[0]}
echo "[eval] perf exit=${PERF_STATUS}"

# ---- 13. HumanEval pass@1 ----
echo "[eval] HumanEval generate..."
/usr/bin/python "${TUTORIAL_DIR}/scripts/humaneval_generate.py" \
    --port "${PORT}" --output "${HE_SAMPLES}" 2>&1 | tee "${HE_GEN_LOG}"
HE_GEN_STATUS=${PIPESTATUS[0]}
if [ "${HE_GEN_STATUS}" -eq 0 ]; then
    echo "[eval] HumanEval evaluate..."
    /usr/bin/python "${TUTORIAL_DIR}/scripts/humaneval_evaluate.py" \
        --samples "${HE_SAMPLES}" 2>&1 | tee "${HE_EVAL_LOG}"
    HE_EVAL_STATUS=${PIPESTATUS[0]}
else
    echo "[eval] HumanEval generate failed, skipping evaluate"; HE_EVAL_STATUS=1
fi

# ---- final dispatch totals ----
echo "[eval] === final dispatch totals ==="
grep -nE "v4-dispatch\] TOTALS|FIRST V3 decode HIT|fallback \(first of kind\)|runtime, DISABLING" "${SERVE_LOG}" || true

echo "[eval] DONE perf=${PERF_STATUS} he_gen=${HE_GEN_STATUS} he_eval=${HE_EVAL_STATUS}"
echo "[eval] logs: serve=${SERVE_LOG} perf=${PERF_LOG} he_eval=${HE_EVAL_LOG}"
# cleanup() runs on EXIT
[ "${PERF_STATUS}" -eq 0 ] && [ "${HE_EVAL_STATUS}" -eq 0 ]
