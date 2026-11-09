#!/bin/bash
# Source from Slurm job scripts after setting WORK.
set -euo pipefail
: "${WORK:?set WORK to the ReVA runtime tree}"

if [[ -f "$WORK/venv312/bin/activate" ]]; then
  source "$WORK/venv312/bin/activate"
elif [[ -f "$WORK/envs/reva/bin/activate" ]]; then
  source "$WORK/envs/reva/bin/activate"
else
  echo "ReVA venv missing: expected $WORK/venv312/bin/activate or $WORK/envs/reva/bin/activate" >&2
  exit 1
fi

# Keep every cache off /home/ambpdc: that home is shared and sits at its 42G hard quota.
# A failed quota write left 0-byte entries in ~/.cache/torch/kernels; PyTorch then loaded
# the empty sm_90 `reduction_prod` kernel and every H100/H200 run died with
# "CUDA driver error: invalid argument ... cuModuleLoadData: image cannot be NULL".
export XDG_CACHE_HOME="${REVA_CACHE_DIR:-$WORK/.cache}"
export PYTORCH_KERNEL_CACHE_PATH="$XDG_CACHE_HOME/torch/kernels"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH" "$TRITON_CACHE_DIR"
# Compile NVRTC kernels in memory per process instead of trusting files on disk.
export USE_PYTORCH_KERNEL_CACHE="${USE_PYTORCH_KERNEL_CACHE:-0}"
# Backstop if the cache is ever re-enabled: never let torch load an empty kernel image.
while IFS= read -r -d '' empty; do
  mv -f -- "$empty" "$empty.empty"
  echo "quarantined empty kernel cache entry: $empty" >&2
done < <(find "$PYTORCH_KERNEL_CACHE_PATH" -type f -size 0 ! -name '*.empty' -print0)
