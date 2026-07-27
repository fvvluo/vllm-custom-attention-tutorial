#!/usr/bin/env bash
# =============================================================================
# setup_vllm_source.sh
# -----------------------------------------------------------------------------
# 作用：把 vLLM 源码 clone 下来并 checkout 到教程指定的 commit，然后复用镜像里
#      已经编译好的 CUDA 扩展（.so），这样你可以随意修改 vLLM 的 Python 源码
#      （比如添加自定义 attention backend），而【不需要】重新编译 CUDA。
#
# 背景（重要，务必理解）：
#   本镜像已通过 wheel 安装了 vLLM（含编译好的 .so），版本正好是本教程要求的
#   commit a49d37c6b。我们把源码放到 PYTHONPATH 最前面来“覆盖”已安装的 .py，
#   但 vLLM 的编译扩展是以“包内子模块”方式导入的（import vllm._C_stable_libtorch），
#   Python 只会在 sys.path 里【第一个】vllm 目录下找这些 .so。源码树本身没有 .so，
#   因此必须把已安装包里的所有 .so 软链接到源码树对应位置，才能 import 成功。
#
# 用法：
#   bash scripts/setup_vllm_source.sh
# =============================================================================
set -euo pipefail

# ---- 可按需修改的配置 -------------------------------------------------------
VLLM_COMMIT="a49d37c6b"
VLLM_SRC="${VLLM_SRC:-/dockerdata/landojiang/vllm_src}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
# -----------------------------------------------------------------------------

echo "==> 定位已安装的 vLLM（用于取编译好的 .so）"
INSTALLED_VLLM="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "    已安装 vLLM 路径: ${INSTALLED_VLLM}"
INSTALLED_VER="$(python -c 'import vllm; print(vllm.__version__)')"
echo "    已安装 vLLM 版本: ${INSTALLED_VER}"

echo "==> 克隆 vLLM 源码到 ${VLLM_SRC}"
if [ ! -d "${VLLM_SRC}/.git" ]; then
    git clone "${VLLM_REPO}" "${VLLM_SRC}"
else
    echo "    已存在，跳过 clone"
fi

echo "==> checkout 到 commit ${VLLM_COMMIT}"
git -C "${VLLM_SRC}" fetch --all --tags --quiet || true
git -C "${VLLM_SRC}" checkout "${VLLM_COMMIT}"
echo "    当前 commit: $(git -C "${VLLM_SRC}" rev-parse --short HEAD)"

echo "==> 把已安装包里所有编译产物(.so)软链接到源码树对应位置"
# 遍历已安装 vllm 目录下的每一个 .so，按相对路径在源码树里建同名软链接。
# 覆盖：顶层 *.abi3.so、vllm_flash_attn/*.so、third_party/deep_gemm/*.so 等。
count=0
while IFS= read -r -d '' so; do
    rel="${so#${INSTALLED_VLLM}/}"          # 相对 vllm 包的路径
    dst="${VLLM_SRC}/vllm/${rel}"
    mkdir -p "$(dirname "${dst}")"
    ln -sf "${so}" "${dst}"
    count=$((count + 1))
done < <(find "${INSTALLED_VLLM}" -name '*.so' -print0)
echo "    已软链接 ${count} 个 .so"

# _version.py 是构建时生成的（含 commit 哈希），源码树里没有；软链过来，
# 这样 vllm.__version__ 会正确显示 0.18.1rc1.dev3933+ga49d37c6b，也消除告警。
if [ -f "${INSTALLED_VLLM}/_version.py" ]; then
    ln -sf "${INSTALLED_VLLM}/_version.py" "${VLLM_SRC}/vllm/_version.py"
    echo "    已软链接 _version.py"
fi

echo "==> 校验：确认 import 的 vllm 来自源码树，且编译扩展可用"
PYTHONPATH="${VLLM_SRC}:${PYTHONPATH:-}" python - <<PYEOF
import vllm
assert "${VLLM_SRC}" in vllm.__file__, f"vllm 未从源码树加载: {vllm.__file__}"
print("  vllm.__file__ =", vllm.__file__)
import vllm._C_stable_libtorch  # 关键：编译扩展必须能 import
print("  vllm._C_stable_libtorch: OK")
import torch
print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
print("  vllm.__version__:", vllm.__version__)
print("SETUP OK")
PYEOF

echo ""
echo "============================================================"
echo "setup 完成。后续启动/运行请务必带上："
echo "  export PYTHONPATH=${VLLM_SRC}:\$PYTHONPATH"
echo "============================================================"
