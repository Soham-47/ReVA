#!/bin/bash
# Round-1 smoke: 20 steps, checkpoint resume check, prompt dump, 20-valQ score.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J reva_smoke_r1
#SBATCH -o reva_smoke_round1_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
# shellcheck source=/dev/null
source "$WORK/jobs/activate_reva.sh"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"

python code/smoke.py

smoke_arm() {
  local cfg=$1
  local score_extra=("${@:2}")
  local out
  out="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$cfg")"
  rm -rf "$out"
  mkdir -p "$out"
  echo "=== prompts $cfg ==="
  python tools/print_round1_prompts.py "$cfg" --samples 5
  echo "=== train phase 1 (10 steps then STOP) $cfg ==="
  python code/train.py "$cfg" 2>&1 | tee "$out/train.log" &
  local pid=$!
  local waited=0
  while ((waited < 7200)); do
    if [[ -d "$out/checkpoint-10" ]]; then
      touch "$out/STOP"
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 10
    waited=$((waited + 10))
  done
  wait "$pid" || true
  rm -f "$out/STOP"
  echo "=== train phase 2 (resume to max_steps) $cfg ==="
  python code/train.py "$cfg" 2>&1 | tee -a "$out/train.log"
  local ckpt
  ckpt="$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  cp "$out/run_meta.json" "$ckpt/run_meta.json"
  grep -c max_tokens_downsample "$out/train.log" | tee "$out/downsample_count.txt" || true
  echo "=== score valQ shard 0/30 (~20 Q) $cfg adapter=$ckpt ==="
  python code/score.py --adapter "$ckpt" --split valQ --out "$out/scores_valQ_smoke20.jsonl" \
    --shard 0 --nshards 30 "${score_extra[@]}"
}

smoke_arm code/configs/smoke_b1_hint16.yaml --motion-hints data/camera_motion_hints.json
smoke_arm code/configs/smoke_b2_frames32.yaml --frame-budget 32 --video-max-pixels 409600
echo "smoke_round1 complete at $(date -Is)"
