#!/bin/bash
# Download and prepare the pinned Charades-STA pilot data on CPU nodes.
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -J reva_charades_data
#SBATCH -o reva_charades_data_%j.out
set -euo pipefail

: "${WORK:=/scratch/ambpdc/soham/ReVA}"
source "$WORK/envs/reva/bin/activate"
export HF_HOME="$WORK/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$WORK"

dataset_parent="$WORK/data/Charades-STA"
dataset_root="$dataset_parent/charades_sta"
run_dir="$WORK/runs/charades_grounding"
mkdir -p "$dataset_root" "$WORK/logs" "$run_dir"
job_id="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$WORK/logs/charades_data_${job_id}.log") 2>&1

if [[ ! -f "$run_dir/data/manifest.json" ]]; then
  if [[ -d "$run_dir/data" ]] && find "$run_dir/data" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite incomplete pilot data output: $run_dir/data" >&2
    exit 2
  fi
  hf download yeliudev/VideoMind-Dataset \
    charades_sta/charades_sta_train.txt \
    charades_sta/charades_sta_test.txt \
    charades_sta/durations.json \
    charades_sta/videos_3fps_480_noaudio.tar.gz \
    --repo-type dataset \
    --revision 3518d8e8c2c7c1ceec4b242afc07162bbafa33d8 \
    --local-dir "$dataset_parent"

  archive="$dataset_root/videos_3fps_480_noaudio.tar.gz"
  if [[ ! -f "$archive" ]]; then
    echo "download completed without the pinned video archive: $archive" >&2
    exit 2
  fi
  if ! find "$dataset_root/videos_3fps_480_noaudio" -type f -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
    tar -xzf "$archive" -C "$dataset_root"
  fi
  if ! find "$dataset_root/videos_3fps_480_noaudio" -type f -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
    echo "archive extraction did not create videos_3fps_480_noaudio/*.mp4" >&2
    exit 2
  fi
  python code/charades_grounding.py prepare \
    --dataset-root "$dataset_root" \
    --output-dir "$run_dir/data" \
    --sample-rows 1024 --dev-fraction 0.1 --seed 0
fi

echo "Charades-STA data ready: $run_dir/data/manifest.json"
cat "$run_dir/data/manifest.json"
