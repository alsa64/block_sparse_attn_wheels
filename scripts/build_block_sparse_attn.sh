#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_attention_common.sh"

BSA_CLONE_DIR="${BSA_CLONE_DIR:-/tmp/Block-Sparse-Attention}"
BSA_GIT_URL="https://github.com/mit-han-lab/Block-Sparse-Attention.git"
BSA_GIT_REF="${BSA_GIT_REF:-}"

init_build_env

# Check if already installed
if "$VENV_PYTHON" -c "import block_sparse_attn; print(f'v{block_sparse_attn.__version__}')" 2>/dev/null; then
	info "Block-Sparse Attention is already installed."
	info "To force rebuild, uninstall first: uv pip uninstall block-sparse-attn"
	exit 0
fi

resolve_cuda_archs
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="$ATTN_CUDA_ARCHS"

info "Cloning Block-Sparse Attention..."
clone_or_update_repo "$BSA_CLONE_DIR" "$BSA_GIT_URL" "$BSA_GIT_REF"
cd "$BSA_CLONE_DIR"

info "Initialising cutlass submodule..."
git submodule update --init csrc/cutlass

ensure_build_deps
resolve_max_jobs

info "Building Block-Sparse Attention wheel (this may take 5-15 minutes)..."
info "  CUDA_HOME=$CUDA_HOME"
info "  BLOCK_SPARSE_ATTN_CUDA_ARCHS=$BLOCK_SPARSE_ATTN_CUDA_ARCHS"
info "  MAX_JOBS=$MAX_JOBS"

cd "$BSA_CLONE_DIR"

clean_python_build_artifacts

"$VENV_PYTHON" -m build --wheel --no-isolation

BSA_WHEEL=$(
	"$VENV_PYTHON" - <<'PY'
from pathlib import Path

wheels = sorted(Path("dist").glob("block_sparse_attn-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
print(wheels[0] if wheels else "")
PY
)

if [ -z "$BSA_WHEEL" ]; then
	error "Wheel build failed - no .whl file found in $BSA_CLONE_DIR/dist/"
	error "Check the full build output above for errors."
	exit 1
fi

info "Built wheel: $BSA_WHEEL"

if [ "${BSA_SKIP_INSTALL:-1}" = "1" ]; then
	info "Skipping install (BSA_SKIP_INSTALL=1)."
	exit 0
fi

cd "$PROJECT_DIR"
info "Installing wheel into venv..."
uv pip install --python "$VENV_PYTHON" "$BSA_CLONE_DIR/$BSA_WHEEL"

BSA_VERSION=$("$VENV_PYTHON" -c "import block_sparse_attn; print(block_sparse_attn.__version__)" 2>/dev/null)
if [ -n "$BSA_VERSION" ]; then
	info "Block-Sparse Attention v$BSA_VERSION installed successfully!"
else
	error "Installation verification failed - could not import block_sparse_attn"
	exit 1
fi
