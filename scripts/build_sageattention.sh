#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_attention_common.sh"

SAGEATTENTION_CLONE_DIR="${SAGEATTENTION_CLONE_DIR:-/tmp/SageAttention}"
SAGEATTENTION_GIT_URL="https://github.com/thu-ml/SageAttention.git"
SAGEATTENTION_GIT_REF="${SAGEATTENTION_GIT_REF:-}"

init_build_env
resolve_cuda_archs
ensure_build_deps
resolve_max_jobs

info "Cloning SageAttention..."
clone_or_update_repo "$SAGEATTENTION_CLONE_DIR" "$SAGEATTENTION_GIT_URL" "$SAGEATTENTION_GIT_REF"

cd "$SAGEATTENTION_CLONE_DIR"
clean_python_build_artifacts

export TORCH_CUDA_ARCH_LIST="$ATTN_CUDA_ARCHS"
export SAGEATTENTION_CUDA_ARCHS="$ATTN_CUDA_ARCHS"
export EXT_PARALLEL="$MAX_JOBS"

info "Building SageAttention wheel..."
info "  CUDA_HOME=$CUDA_HOME"
info "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
info "  EXT_PARALLEL=$EXT_PARALLEL"

"$VENV_PYTHON" -m build --wheel --no-isolation

SAGEATTENTION_WHEEL=$(
	"$VENV_PYTHON" - <<'PY'
from pathlib import Path

wheels = sorted(Path("dist").glob("sageattention-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
print(wheels[0] if wheels else "")
PY
)

if [ -z "$SAGEATTENTION_WHEEL" ]; then
	error "Wheel build failed - no .whl file found in $SAGEATTENTION_CLONE_DIR/dist/"
	exit 1
fi

info "Built wheel: $SAGEATTENTION_WHEEL"

if [ "${SAGEATTENTION_SKIP_INSTALL:-1}" = "1" ]; then
	info "Skipping install (SAGEATTENTION_SKIP_INSTALL=1)."
	exit 0
fi

cd "$PROJECT_DIR"
info "Installing wheel into venv..."
uv pip install --python "$VENV_PYTHON" "$SAGEATTENTION_CLONE_DIR/$SAGEATTENTION_WHEEL"

if "$VENV_PYTHON" -c "import sageattention" 2>/dev/null; then
	info "SageAttention installed successfully!"
else
	error "Installation verification failed - could not import sageattention"
	exit 1
fi
