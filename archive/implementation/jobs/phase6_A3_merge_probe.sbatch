#!/bin/bash
# Five-example answer-token parity check for the trained A3 adapter.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH --open-mode=append
#SBATCH -J reva_a3_merge
#SBATCH -o reva_a3_merge_%j.out
set -euo pipefail

: "${WORK:?set WORK to the HPC runtime tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"

python code/adapter_merge_probe.py \
  --out runs/phase6_A3_language_plus_vision_from_stageA/adapter_merge_probe.json \
  --arm A3="$WORK/runs/phase6_A3_language_plus_vision_from_stageA/final_adapter"
