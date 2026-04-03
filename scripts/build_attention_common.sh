#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[ATTN]${NC} $*"; }
warn() { echo -e "${YELLOW}[ATTN]${NC} $*"; }
error() { echo -e "${RED}[ATTN]${NC} $*" >&2; }

init_build_env() {
	cd "$PROJECT_DIR"

	VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
	if [ ! -x "$VENV_PYTHON" ]; then
		error "Venv python not found at $VENV_PYTHON - run 'uv sync' first."
		exit 1
	fi

	export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
	if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
		error "nvcc not found at $CUDA_HOME/bin/nvcc"
		error "Set CUDA_HOME to the CUDA toolkit directory containing bin/nvcc."
		exit 1
	fi

	NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
	info "CUDA toolkit: $CUDA_HOME (nvcc $NVCC_VERSION)"

	TORCH_CUDA=$("$VENV_PYTHON" -c "import torch; print(torch.version.cuda)")
	info "PyTorch CUDA: $TORCH_CUDA"
}

resolve_cuda_archs() {
	local requested_archs="${ATTN_CUDA_ARCHS:-${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-${SAGEATTENTION_CUDA_ARCHS:-${FLASH_ATTN_CUDA_ARCHS:-}}}}"

	if [ -n "$requested_archs" ]; then
		export ATTN_CUDA_ARCHS="$requested_archs"
		info "Using user-specified arch(s): $ATTN_CUDA_ARCHS"
		return
	fi

	ATTN_CUDA_ARCHS=$(
		"$VENV_PYTHON" -c '
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if arch >= 120:
        print("80;90;100;120")
    elif arch >= 100:
        print("80;90;100")
    elif arch >= 90:
        print("80;90")
    else:
        print("80")
else:
    print("80")
'
	)
	export ATTN_CUDA_ARCHS
	info "Auto-detected GPU arch(s): $ATTN_CUDA_ARCHS"
}

ensure_build_deps() {
	cd "$PROJECT_DIR"
	info "Installing build dependencies..."
	uv pip install --python "$VENV_PYTHON" build packaging ninja psutil wheel setuptools
	export PATH="$PROJECT_DIR/.venv/bin:$PATH"

	if "$VENV_PYTHON" -c "from torch.utils.cpp_extension import is_ninja_available; assert is_ninja_available()" 2>/dev/null; then
		NINJA_VERSION=$("$PROJECT_DIR/.venv/bin/ninja" --version 2>/dev/null || echo "?")
		info "Ninja build system: v$NINJA_VERSION"
	else
		error "Ninja not found - required for parallel compilation."
		exit 1
	fi
}

resolve_max_jobs() {
	if [ -z "${MAX_JOBS:-}" ]; then
		MAX_JOBS=$("$VENV_PYTHON" -c "import os; print(min(8, max(2, os.cpu_count() // 8)))")
	fi
	export MAX_JOBS
	info "MAX_JOBS=$MAX_JOBS"
}

clone_or_update_repo() {
	local repo_dir="$1"
	local repo_url="$2"
	local repo_ref="${3:-}"

	if [ -d "$repo_dir/.git" ]; then
		info "Repo already cloned at $repo_dir"
		if [ -n "$repo_ref" ]; then
			git -C "$repo_dir" fetch --depth=1 origin "$repo_ref"
		else
			git -C "$repo_dir" fetch --depth=1 origin
		fi
		git -C "$repo_dir" checkout --detach FETCH_HEAD
	else
		if [ -n "$repo_ref" ]; then
			git clone --depth=1 --branch "$repo_ref" "$repo_url" "$repo_dir"
		else
			git clone --depth=1 "$repo_url" "$repo_dir"
		fi
	fi
}

clean_python_build_artifacts() {
	rm -rf dist/ build/ *.egg-info
}

install_python_packages() {
	cd "$PROJECT_DIR"
	uv pip install --python "$VENV_PYTHON" "$@"
}
