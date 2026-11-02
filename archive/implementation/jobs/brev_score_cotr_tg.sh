#!/usr/bin/env bash
# CoTR visual timestamp burn-in on Temporal Grounding (frozen Stage A).
set -euo pipefail
WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
ADAPTER="${ADAPTER:-$WORK/runs/stageA_q3vl8b_r16/final_adapter}"
OUT_ROOT="${OUT_ROOT:-$WORK/runs/cotr_tg_burnin}"
IDS="${IDS:-$REPO/results/visual_prompt_pilot/reze_tg_ids.json}"
SCORE="${SCORE:-$REPO/archive/implementation/code/score_stageA_pilots.py}"
STALL_SECS="${STALL_SECS:-180}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PY="$VENV/bin/python"
cd "$REPO"
test -f "$ADAPTER/adapter_config.json"
test -f "$SCORE"
mkdir -p "$OUT_ROOT"

n_ids() {
  local ids="$1" split="$2"
  "$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['splits'][sys.argv[2]]['ids']))" "$ids" "$split"
}

score_until_done() {
  local split="$1" out="$2"
  local target
  target="$(n_ids "$IDS" "$split")"
  echo "CoTR TG $split -> $out (target $target)"
  while true; do
    local done=0
    if [[ -f "$out" ]]; then
      done="$(grep -c . "$out" || true)"
    fi
    if [[ "$done" -ge "$target" ]]; then
      echo "$split complete ($done/$target)"
      return 0
    fi
    WORK="$WORK" "$PY" "$SCORE" \
      --adapter "$ADAPTER" \
      --split "$split" \
      --qa-ids-file "$IDS" \
      --out "$out" \
      --cotr-timestamp-burnin &
    local pid=$!
    local last="$done"
    local stall=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 15
      local now=0
      if [[ -f "$out" ]]; then
        now="$(grep -c . "$out" || true)"
      fi
      if [[ "$now" -eq "$last" ]]; then
        stall=$((stall + 15))
      else
        stall=0
        last="$now"
      fi
      if [[ "$stall" -ge "$STALL_SECS" ]]; then
        echo "stall ${stall}s at $now/$target; restarting pid $pid"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        break
      fi
    done
    wait "$pid" 2>/dev/null || true
  done
}

for split in valQ valT; do
  score_until_done "$split" "$OUT_ROOT/${split}_cotr.jsonl"
  BASELINE="$WORK/runs/stageA_q3vl8b_r16/legacy/scores_${split}.jsonl"
  WORK="$WORK" "$PY" archive/implementation/code/analyze_visual_prompt_pilot.py \
    --split "$split" \
    --ids "$IDS" \
    --baseline "$BASELINE" \
    --arm cotr "$OUT_ROOT/${split}_cotr.jsonl" \
    --out "$OUT_ROOT/${split}_report.json"
done
echo "CoTR TG burn-in finished -> $OUT_ROOT"
