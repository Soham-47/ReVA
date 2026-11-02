#!/bin/bash
# Charades interval SFT/GRPO + paired ReVA eval on H100 (isolated run dir).
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_charades_h100
#SBATCH -o reva_charades_h100_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
export REVA_GPU_TAG=h100
export REVA_RUN_DIR="${REVA_RUN_DIR:-$WORK/runs/charades_grounding_h100}"
bash "$WORK/jobs/charades_grounding_train_body.sh"
