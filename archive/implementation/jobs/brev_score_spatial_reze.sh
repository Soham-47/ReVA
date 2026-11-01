#!/usr/bin/env bash
# CROSS spatial-contract then REZE TG fusion on frozen Stage A.
set -euo pipefail
WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
ADAPTER="${ADAPTER:-$WORK/runs/stageA_q3vl8b_r16/final_adapter}"
SPATIAL_OUT="${SPATIAL_OUT:-$WORK/runs/spatial_contract}"
REZE_OUT="${REZE_OUT:-$WORK/runs/reze_tg}"
SPATIAL_IDS="${SPATIAL_IDS:-$REPO/results/visual_prompt_pilot/spatial_contract_ids.json}"
REZE_IDS="${REZE_IDS:-$REPO/results/visual_prompt_pilot/reze_tg_ids.json}"
STALL_SECS="${STALL_SECS:-180}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PY="$VENV/bin/python"
cd "$REPO"
test -f "$ADAPTER/adapter_config.json"
mkdir -p "$SPATIAL_OUT" "$REZE_OUT"

n_ids() {
  local ids="$1" split="$2"
  "$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['splits'][sys.argv[2]]['ids']))" "$ids" "$split"
}

score_until_done() {
  local split="$1" ids="$2" out="$3"
  shift 3
  local target
  target="$(n_ids "$ids" "$split")"
  echo "scoring $split -> $out (target $target)"
  while true; do
    local done=0
    if [[ -f "$out" ]]; then
      done="$(grep -c . "$out" || true)"
    fi
    if [[ "$done" -ge "$target" ]]; then
      echo "$split complete ($done/$target)"
      return 0
    fi
    WORK="$WORK" "$PY" code/score.py \
      --adapter "$ADAPTER" \
      --split "$split" \
      --qa-ids-file "$ids" \
      --out "$out" \
      "$@" &
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
  out="$SPATIAL_OUT/${split}_contract.jsonl"
  score_until_done "$split" "$SPATIAL_IDS" "$out" --spatial-contract
  WORK="$WORK" "$PY" code/analyze_visual_prompt_pilot.py \
    --split "$split" \
    --ids "$SPATIAL_IDS" \
    --baseline "$WORK/runs/stageA_q3vl8b_r16/legacy/scores_${split}.jsonl" \
    --arm contract "$out" \
    --out "$SPATIAL_OUT/${split}_report.json"
done

for split in valQ valT; do
  out="$REZE_OUT/${split}_reze.jsonl"
  score_until_done "$split" "$REZE_IDS" "$out" --reze --reze-alpha 1.0
done
WORK="$WORK" "$PY" code/analyze_reze.py \
  --ids "$REZE_IDS" \
  --baseline-valQ "$WORK/runs/stageA_q3vl8b_r16/legacy/scores_valQ.jsonl" \
  --baseline-valT "$WORK/runs/stageA_q3vl8b_r16/legacy/scores_valT.jsonl" \
  --reze-valQ "$REZE_OUT/valQ_reze.jsonl" \
  --reze-valT "$REZE_OUT/valT_reze.jsonl" \
  --out "$REZE_OUT/report.json"
echo "spatial-contract and REZE scoring finished"
