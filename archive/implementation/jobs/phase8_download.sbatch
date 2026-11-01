#!/bin/bash
# Pre-download the two pinned Phase 8 backbone snapshots into WORK's cache.
# This is CPU-only and writes no model files or results outside hf_cache.
# The scoring job sets offline mode and must run only after this job succeeds.
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J reva_phase8_download
#SBATCH -o reva_phase8_download_%j.out
set -euo pipefail

: "${WORK:?set WORK to the prepared ReVA runtime tree}"
env_activate="${REVA_ENV:-$WORK/envs/reva}"
if [[ -d "$env_activate" ]]; then
  env_activate="$env_activate/bin/activate"
fi
if [[ ! -f "$env_activate" ]]; then
  echo "missing prepared Python environment: $env_activate" >&2
  exit 2
fi
source "$env_activate"

python_bin="${REVA_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
cd "$WORK"

job_id="${SLURM_JOB_ID:-manual}"
mkdir -p "$WORK/logs"
exec > >(tee -a "$WORK/logs/phase8_download_${job_id}.log") 2>&1

HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" "$python_bin" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

SNAPSHOTS = (
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ("Qwen/Qwen3-VL-32B-Instruct", "0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
)

cache_dir = Path(os.environ["HF_HUB_CACHE"]).resolve()
for repo_id, revision in SNAPSHOTS:
    print(f"downloading {repo_id}@{revision} into {cache_dir}", flush=True)
    path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=str(cache_dir))
    resolved = Path(path).resolve()
    if cache_dir not in resolved.parents:
        raise RuntimeError(f"snapshot escaped HF_HUB_CACHE: {resolved}")
    print(f"ready: {resolved}", flush=True)
PY

# Resolve both processor/config records offline through the same cache root
# used by phase8_score.sbatch.  This catches an incomplete or misplaced cache
# before an H200 allocation is spent on model loading.
HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$python_bin" - <<'PY'
import os

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

SNAPSHOTS = (
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ("Qwen/Qwen3-VL-32B-Instruct", "0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
)
cache_dir = os.environ["HF_HUB_CACHE"]
for repo_id, revision in SNAPSHOTS:
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    AutoConfig.from_pretrained(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    AutoProcessor.from_pretrained(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    print(f"verified offline: {repo_id}@{revision} -> {path}", flush=True)
PY

echo "Phase 8 pinned backbone snapshots are ready in $HF_HUB_CACHE"
