#!/bin/bash
# Round-1 smoke (same logic as smoke_round1.sbatch). Usage: bash jobs/smoke_round1_body.sh
set -euo pipefail

: "${WORK:?set WORK}"
# shellcheck source=/dev/null
source "$WORK/jobs/activate_reva.sh"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"

python code/smoke.py

smoke_arm() {
  local cfg=$1
  shift
  local score_extra=("$@")
  local out
  out="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$cfg")"
  rm -rf "$out"
  mkdir -p "$out"
  echo "=== prompts $cfg ==="
  python tools/print_round1_prompts.py "$cfg" --samples 5 | tee "$out/prompts.txt"
  echo "=== train phase 1 (until checkpoint-10, then STOP) $cfg ==="
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
  if [[ ! -d "$out/checkpoint-10" ]]; then
    echo "FAIL: no checkpoint-10 under $out" >&2
    return 1
  fi
  echo "=== train phase 2 (resume to max_steps) $cfg ==="
  python code/train.py "$cfg" 2>&1 | tee -a "$out/train.log"
  local ckpt
  ckpt="$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  cp "$out/run_meta.json" "$ckpt/run_meta.json"
  local downsample_count
  downsample_count="$(grep -c max_tokens_downsample "$out/train.log" || true)"
  echo "max_tokens_downsample_count=$downsample_count" | tee "$out/downsample_count.txt"
  echo "=== score valQ shard 0/30 (~20 Q) $cfg adapter=$ckpt ==="
  python code/score.py --adapter "$ckpt" --split valQ --out "$out/scores_valQ_smoke20.jsonl" \
    --shard 0 --nshards 30 "${score_extra[@]}"
  echo "=== scored $(wc -l < "$out/scores_valQ_smoke20.jsonl") lines ==="
}

smoke_arm code/configs/smoke_b1_hint16.yaml --motion-hints data/camera_motion_hints.json
smoke_arm code/configs/smoke_b2_frames32.yaml --frame-budget 32 --video-max-pixels 409600
echo "smoke_round1 complete at $(date -Is)"
