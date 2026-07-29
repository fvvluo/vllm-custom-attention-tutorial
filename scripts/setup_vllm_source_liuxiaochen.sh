#!/usr/bin/env bash
# =============================================================================
# setup_vllm_source_liuxiaochen.sh
# -----------------------------------------------------------------------------
# Personalized copy of setup_vllm_source.sh (author: landojiang) for Liu Xiaochen.
# ONLY change vs the original: path parameterization + extra safety guards.
# Semantics are unchanged: clone vLLM source, checkout the tutorial commit, and
# symlink the already-compiled .so extensions from the installed wheel into the
# source tree so `import vllm._C_stable_libtorch` works without recompiling.
#
# It does NOT: delete existing dirs, pip install/uninstall vLLM, compile CUDA,
# write to ~/.bashrc. All symlinks live only under $VLLM_SRC.
# =============================================================================
set -euo pipefail

# ---- personal, env-overridable configuration ----
TUTORIAL_DIR="${TUTORIAL_DIR:-/dockerdata/liuxiaochen/vllm_tutorial}"
VLLM_SRC="${VLLM_SRC:-/dockerdata/liuxiaochen/vllm_src}"
VLLM_COMMIT="${VLLM_COMMIT:-a49d37c6b}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
# -------------------------------------------------

echo "==> config"
echo "    TUTORIAL_DIR=${TUTORIAL_DIR}"
echo "    VLLM_SRC=${VLLM_SRC}"
echo "    VLLM_COMMIT=${VLLM_COMMIT}"
echo "    PYTHON_BIN=${PYTHON_BIN}"

echo "==> locate installed vLLM (source of compiled .so)"
INSTALLED_VLLM="$("${PYTHON_BIN}" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
INSTALLED_VER="$("${PYTHON_BIN}" -c 'import vllm; print(vllm.__version__)')"
echo "    installed vLLM path: ${INSTALLED_VLLM}"
echo "    installed vLLM ver : ${INSTALLED_VER}"

# ---- guard: existing VLLM_SRC ----
if [ -e "${VLLM_SRC}" ]; then
    if [ -d "${VLLM_SRC}/.git" ]; then
        cur="$(git -C "${VLLM_SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "==> ${VLLM_SRC} already a git repo at ${cur}"
        # Refuse to touch a dirty tree (protect any in-progress edits).
        if [ -n "$(git -C "${VLLM_SRC}" status --porcelain 2>/dev/null)" ]; then
            echo "ERROR: ${VLLM_SRC} has local modifications; refusing to checkout. Inspect first." >&2
            exit 3
        fi
    elif [ -n "$(ls -A "${VLLM_SRC}" 2>/dev/null)" ]; then
        echo "ERROR: ${VLLM_SRC} exists, is non-empty and not a git repo; refusing to overwrite." >&2
        exit 3
    fi
fi

echo "==> clone vLLM source to ${VLLM_SRC} (skip if present)"
if [ ! -d "${VLLM_SRC}/.git" ]; then
    git clone "${VLLM_REPO}" "${VLLM_SRC}"
else
    echo "    already present, skipping clone"
fi

echo "==> checkout commit ${VLLM_COMMIT}"
git -C "${VLLM_SRC}" fetch --all --tags --quiet || true
git -C "${VLLM_SRC}" checkout "${VLLM_COMMIT}"
ACTUAL="$(git -C "${VLLM_SRC}" rev-parse HEAD)"
echo "    current commit: ${ACTUAL}"
case "${ACTUAL}" in
    ${VLLM_COMMIT}*) : ;;
    *)
        # a49d37c6b is a short hash; verify it is an ancestor / matches.
        if ! git -C "${VLLM_SRC}" merge-base --is-ancestor "${VLLM_COMMIT}" HEAD 2>/dev/null \
           && [ "$(git -C "${VLLM_SRC}" rev-parse --short=9 HEAD)" != "${VLLM_COMMIT}" ]; then
            echo "ERROR: checked-out commit ${ACTUAL} != requested ${VLLM_COMMIT}; stopping." >&2
            exit 4
        fi
        ;;
esac

echo "==> symlink all compiled .so from installed wheel into the source tree"
count=0
while IFS= read -r -d '' so; do
    rel="${so#${INSTALLED_VLLM}/}"
    dst="${VLLM_SRC}/vllm/${rel}"
    mkdir -p "$(dirname "${dst}")"
    ln -sf "${so}" "${dst}"
    echo "    link: ${dst} -> ${so}"
    count=$((count + 1))
done < <(find "${INSTALLED_VLLM}" -name '*.so' -print0)
echo "    symlinked ${count} .so files"

if [ -f "${INSTALLED_VLLM}/_version.py" ]; then
    ln -sf "${INSTALLED_VLLM}/_version.py" "${VLLM_SRC}/vllm/_version.py"
    echo "    link: ${VLLM_SRC}/vllm/_version.py -> ${INSTALLED_VLLM}/_version.py"
fi

echo "==> verify: import vllm from source tree + compiled extension usable"
PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<PYEOF
import vllm
assert "${VLLM_SRC}" in vllm.__file__, f"vllm not from source tree: {vllm.__file__}"
print("  vllm.__file__ =", vllm.__file__)
import vllm._C_stable_libtorch
print("  vllm._C_stable_libtorch: OK")
import torch
print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
print("  vllm.__version__:", vllm.__version__)
print("SETUP OK")
PYEOF

echo ""
echo "============================================================"
echo "setup done. Use:  export PYTHONPATH=${VLLM_SRC}:\$PYTHONPATH"
echo "============================================================"
