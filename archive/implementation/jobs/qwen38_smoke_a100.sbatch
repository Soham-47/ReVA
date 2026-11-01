#!/bin/bash
# Isolated Qwen3.8 finite smoke on A100. This never starts full training.
# The config is Stage A-equivalent; only output_dir/run_name isolate the smoke.
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH --open-mode=append
#SBATCH -J reva_qwen38_a100_smoke
#SBATCH -o reva_qwen38_a100_smoke_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch jobs/qwen38_smoke_a100.sbatch code/configs/stageA_qwen38_a100_smoke.yaml" >&2
  exit 2
fi
CFG=$1
if [[ "$CFG" != *qwen38_a100_smoke* ]]; then
  echo "A100 smoke wrapper requires the isolated A100 Qwen3.8 config: $CFG" >&2
  exit 2
fi
: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"
OUT="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CFG")"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ -f "$OUT/SMOKE_OK" ]]; then
  echo "smoke already passed: $OUT/SMOKE_OK"
  exit 0
fi
python code/smoke.py --config "$CFG"
touch "$OUT/SMOKE_OK"
echo "smoke-only mode complete; training was not started"
