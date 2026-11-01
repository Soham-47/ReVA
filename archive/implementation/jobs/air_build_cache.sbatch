#!/bin/bash
# Build the exact 64-candidate AIR feature cache before AIR scoring.
# This is val-only and writes only under the supplied AIR cache directory.
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --job-name=reva_air_cache
#SBATCH --output=reva_air_cache_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch jobs/air_build_cache.sbatch val|valQ|valT" >&2
  exit 2
fi
split="$1"
case "$split" in
  val|valQ|valT) ;;
  test) echo "AIR cache builder is val-only; refusing test" >&2; exit 2 ;;
  *) echo "unsupported split: $split" >&2; exit 2 ;;
esac
: "${WORK:?set WORK to the prepared ReVA runtime tree}"
source "${REVA_ENV:-$WORK/envs/reva}/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$WORK"
python code/build_air_feature_cache.py \
  --split "$split" \
  --model-dir "${AIR_SIGLIP_MODEL_DIR:-$WORK/hf_cache/siglip2-base-patch16-224-022b6f7}" \
  --feature-cache-dir "${AIR_FEATURE_CACHE:-$WORK/runs/air_feature_cache}" \
  --count 64 --batch-size "${AIR_CACHE_BATCH_SIZE:-16}" --device cuda
