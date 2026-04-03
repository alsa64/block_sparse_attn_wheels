#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_attention_common.sh"

FLASH_ATTN_CLONE_DIR="${FLASH_ATTN_CLONE_DIR:-/tmp/flash-attention}"
FLASH_ATTN_GIT_URL="https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTN_GIT_REF="${FLASH_ATTN_GIT_REF:-}"

init_build_env
resolve_cuda_archs
ensure_build_deps
resolve_max_jobs

info "Cloning FlashAttention..."
clone_or_update_repo "$FLASH_ATTN_CLONE_DIR" "$FLASH_ATTN_GIT_URL" "$FLASH_ATTN_GIT_REF"

cd "$FLASH_ATTN_CLONE_DIR"
info "Initialising cutlass submodule..."
git submodule update --init csrc/cutlass
clean_python_build_artifacts

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="$ATTN_CUDA_ARCHS"
export NVCC_THREADS="4"

info "Building FlashAttention wheel..."
info "  CUDA_HOME=$CUDA_HOME"
info "  FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
info "  MAX_JOBS=$MAX_JOBS"
info "  NVCC_THREADS=$NVCC_THREADS"

"$VENV_PYTHON" -m build --wheel --no-isolation

FLASH_ATTN_WHEEL=$(
	"$VENV_PYTHON" - <<'PY'
from pathlib import Path

wheels = sorted(Path("dist").glob("flash_attn-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
print(wheels[0] if wheels else "")
PY
)

if [ -z "$FLASH_ATTN_WHEEL" ]; then
	error "Wheel build failed - no .whl file found in $FLASH_ATTN_CLONE_DIR/dist/"
	exit 1
fi

info "Built wheel: $FLASH_ATTN_WHEEL"

if [ "${FLASH_ATTN_SKIP_INSTALL:-1}" = "1" ]; then
	info "Skipping install (FLASH_ATTN_SKIP_INSTALL=1)."
	exit 0
fi

cd "$PROJECT_DIR"
info "Installing wheel into venv..."
uv pip install --python "$VENV_PYTHON" "$FLASH_ATTN_CLONE_DIR/$FLASH_ATTN_WHEEL"

FLASH_ATTN_VERSION=$("$VENV_PYTHON" -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || true)
if [ -n "$FLASH_ATTN_VERSION" ]; then
	info "FlashAttention v$FLASH_ATTN_VERSION installed successfully!"
else
	error "Installation verification failed - could not import flash_attn"
	exit 1
fi
