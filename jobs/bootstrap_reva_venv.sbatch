#!/bin/bash
# One-time (re)build of $WORK/venv312 on HPC when envs/reva/bin is missing.
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J reva_bootstrap_venv
#SBATCH -o reva_bootstrap_venv_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
KPY="${REVA_BOOTSTRAP_PYTHON:-/scratch/ambpdc/soham/kaggle-k12/env/bin/python}"
cd "$WORK"

if [[ ! -x "$KPY" ]]; then
  echo "missing bootstrap python: $KPY" >&2
  exit 1
fi

"$KPY" -m venv "$WORK/venv312"
# shellcheck source=/dev/null
source "$WORK/venv312/bin/activate"
export HF_HOME="$WORK/hf_cache"
pip install -U pip wheel
pip install -r requirements-torch.txt -r requirements.txt opencv-python-headless
python -c "import torch, transformers, peft, av, cv2; print('venv_ok', torch.__version__, transformers.__version__)"
