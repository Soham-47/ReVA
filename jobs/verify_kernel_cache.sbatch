#!/bin/bash
# Short GPU check that the Hopper cuModuleLoadData fix holds (see tools/verify_kernel_cache_fix.py).
# Submit per GPU type, then make training depend on it, e.g.:
#   v=$(sbatch --parsable -p gpu_h200_8 -c 4 --gres=gpu:nvidia_h200_nvl:1 jobs/verify_kernel_cache.sbatch)
#   sbatch --dependency=afterok:$v ... jobs/train.sbatch code/configs/b1_hint16.yaml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J reva_kcache_verify
#SBATCH -o reva_kcache_verify_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
cd "$WORK"
source "$WORK/jobs/activate_reva.sh"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "USE_PYTORCH_KERNEL_CACHE=$USE_PYTORCH_KERNEL_CACHE PYTORCH_KERNEL_CACHE_PATH=$PYTORCH_KERNEL_CACHE_PATH"
python -u tools/verify_kernel_cache_fix.py
