#!/bin/bash
# Isolated H200 correctness/performance smoke for optional Qwen linear-attention kernels.
# Uses the same model, sample set, training targets, and LoRA configuration as baseline smoke.
# This job runs only the bounded forward/backward smoke; it never starts training.
# Submit with a dependency on the baseline H200 smoke to keep comparisons ordered.
#
# Required environment:
#   WORK=/scratch/ambpdc/soham/ReVA
#   KERNEL_SITE=$WORK/envs/reva_qwen38_kernel_site
#
# Example:
#   sbatch --dependency=afterok:<baseline-job> --export=ALL,WORK=...,KERNEL_SITE=... \
#     jobs/qwen38_kernel_smoke_h200.sbatch code/configs/stageA_qwen38_h200_fla_smoke.yaml

#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J reva_qwen38_h200_fla_smoke
#SBATCH -o reva_qwen38_h200_fla_smoke_%j.out

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch jobs/qwen38_kernel_smoke_h200.sbatch code/configs/stageA_qwen38_h200_fla_smoke.yaml" >&2
  exit 2
fi

: "${WORK:?set WORK to the runtime Lustre tree}"
: "${KERNEL_SITE:?set KERNEL_SITE to the isolated optimized-kernel package directory}"
CFG=$1

source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$KERNEL_SITE${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK"

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("flash-linear-attention", "fla-core"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError as exc:
        raise SystemExit(f"required optimized kernel package is missing: {package}") from exc

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except Exception as exc:
    raise SystemExit(f"flash-linear-attention import failed: {exc}") from exc

try:
    import causal_conv1d
    print("causal_conv1d=available")
except Exception as exc:
    print(f"causal_conv1d=unavailable ({exc}); this trial still measures FLA alone")
PY

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
OUT="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CFG")"
mkdir -p "$OUT"
OPT_LOG="$OUT/kernel_smoke_${SLURM_JOB_ID:-manual}.log"
python code/smoke.py --config "$CFG" 2>&1 | tee "$OPT_LOG"

# A successful forward/backward is not enough: fail this optimization trial if
# Transformers silently selected the slow reference PyTorch implementation.
if grep -q 'chunk_gated_delta_rule.*falling back to its reference PyTorch implementation' "$OPT_LOG"; then
  echo "FLA_DISPATCH_VERIFIED=0; optimized kernel was not selected" >&2
  exit 1
fi
echo "FLA_DISPATCH_VERIFIED=1"
touch "$OUT/SMOKE_OK"
echo "optimized-kernel smoke passed: $OUT"
