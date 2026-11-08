#!/bin/bash
# Submitted after user approval 2026-11-05.
# Fail-fast (~15 min) Qwen3-VL video forward probe for the venv312 CUDA bug.
# Does NOT cancel pending job 432730 (reva_b2_h100). Leave it alone.
#
# Suggested submit (only after user approval):
#   WORK=/scratch/ambpdc/soham/ReVA sbatch jobs/cuda_probe.sbatch
#
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -J reva_cuda_probe
#SBATCH -o reva_cuda_probe_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
cd "$WORK"

if [[ ! -x "$WORK/venv312/bin/python" ]]; then
  echo "missing $WORK/venv312/bin/python" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$WORK/jobs/activate_reva.sh" 2>/dev/null || true
# shellcheck source=/dev/null
source "$WORK/venv312/bin/activate"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
# Overridable for A/B diagnosis: sbatch --export=ALL,CUDA_MODULE_LOADING=LAZY,CUDA_LAUNCH_BLOCKING=0 ...
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export REVA_PROBE_OUT="$WORK/tmp_cuda_probe_${SLURM_JOB_ID:-manual}"

mkdir -p "$REVA_PROBE_OUT"
echo "$(date -Is) starting CUDA probe; out=$REVA_PROBE_OUT"
echo "NOTE: pending Slurm 432730 (reva_b2_h100) was NOT cancelled by this draft."
rc=0
python -u "$WORK/tools/cuda_probe_forward.py" || rc=$?
echo "$(date -Is) probe finished rc=$rc"
exit $rc
