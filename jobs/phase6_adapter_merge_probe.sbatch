#!/bin/bash
# Bounded adapter merge parity check; this job does not train or overwrite scores.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH --open-mode=append
#SBATCH -J reva_merge_probe
#SBATCH -o reva_merge_probe_%j.out
set -euo pipefail

: "${WORK:?set WORK to the HPC runtime tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"

python code/adapter_merge_probe.py \
  --out runs/phase6_A0_A1_A2_merge_probe.json \
  --arm A0="$WORK/runs/stageA_q3vl8b_r16/final_adapter" \
  --arm A1="$WORK/runs/phase6_A1_language_from_stageA/final_adapter" \
  --arm A2="$WORK/runs/phase6_A2_h100_train/final_adapter"
