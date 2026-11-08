#!/bin/bash
# Approved setup task: 50-step StepBenchmark for 16f default vs 32f raised pixel cap.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J reva_bench_b1b2
#SBATCH -o reva_bench_b1b2_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
# shellcheck source=/dev/null
source "$WORK/jobs/activate_reva.sh"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/smoke.py

run_arm() {
  local cfg=$1
  local out
  out="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$cfg")"
  mkdir -p "$out"
  if [[ -f "$out/DONE" ]]; then
    echo "skip complete $out"
    return 0
  fi
  echo "=== benchmark arm $cfg -> $out ==="
  /usr/bin/time -f "wall_seconds=%e max_rss_kb=%M" python code/train.py "$cfg" 2>&1 | tee "$out/benchmark.log"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

run_arm code/configs/benchmark_16f.yaml
run_arm code/configs/benchmark_32f.yaml
echo "benchmark job complete at $(date -Is)"
