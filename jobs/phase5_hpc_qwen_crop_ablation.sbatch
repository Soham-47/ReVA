#!/bin/bash
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J reva_phase5_qwen
#SBATCH -o runs/phase5_crop_ablation/logs/phase5_%j.out
#SBATCH -e runs/phase5_crop_ablation/logs/phase5_%j.err
set -euo pipefail

REPO=/scratch/ambpdc/soham/ReVA
PHASE5="$REPO/runs/phase5_crop_ablation"
ENV="$REPO/envs/reva"
export HF_HOME="$REPO/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

source "$ENV/bin/activate"
cd "$REPO"
mkdir -p "$PHASE5/logs" "$PHASE5/results_hpc"
python jobs/phase5_qwen_crop_ablation_hpc.py \
  --input-dir "$PHASE5/input" \
  --output-dir "$PHASE5/results_hpc"
