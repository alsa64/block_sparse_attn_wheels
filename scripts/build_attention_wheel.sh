#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
	echo "usage: $0 <block_sparse_attn|sageattention|flash_attn>" >&2
	exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
block_sparse_attn)
	exec "$SCRIPT_DIR/build_block_sparse_attn.sh"
	;;
sageattention)
	exec "$SCRIPT_DIR/build_sageattention.sh"
	;;
flash_attn)
	exec "$SCRIPT_DIR/build_flash_attn.sh"
	;;
*)
	echo "unknown package: $1" >&2
	exit 2
	;;
esac
