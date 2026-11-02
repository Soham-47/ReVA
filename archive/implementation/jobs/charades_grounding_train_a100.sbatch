#!/bin/bash
# Charades interval SFT/GRPO + paired ReVA eval on A100 (isolated run dir).
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_charades_a100
#SBATCH -o reva_charades_a100_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
export REVA_GPU_TAG=a100
export REVA_RUN_DIR="${REVA_RUN_DIR:-$WORK/runs/charades_grounding_a100}"
bash "$WORK/jobs/charades_grounding_train_body.sh"
