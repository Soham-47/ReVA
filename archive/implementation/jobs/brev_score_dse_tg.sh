#!/usr/bin/env bash
# DSE-VTG TG scoring on Brev: Embedding-8B similarity + gated fusion over Stage A logits.
set -euo pipefail
WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
IDS="${IDS:-$REPO/results/visual_prompt_pilot/reze_tg_ids.json}"
OUT_ROOT="${OUT_ROOT:-$WORK/runs/dse_vtg_brev}"
BASELINE_ROOT="${BASELINE_ROOT:-$WORK/runs/stageA_q3vl8b_r16/legacy}"
STALL_SECS="${STALL_SECS:-240}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PY="$VENV/bin/python"
cd "$REPO"
mkdir -p "$OUT_ROOT"

n_ids() {
  local split="$1"
  "$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['splits'][sys.argv[2]]['ids']))" "$IDS" "$split"
}

score_until_done() {
  local split="$1"
  local out="$2"
  local target
  target="$(n_ids "$split")"
  echo "DSE-VTG $split -> $out (target $target)"
  while true; do
    local done=0
    [[ -f "$out" ]] && done="$(grep -c . "$out" || true)"
    if [[ "$done" -ge "$target" ]]; then
      echo "$split complete ($done/$target)"
      return 0
    fi
    WORK="$WORK" "$PY" archive/implementation/code/score_dse_vtg.py \
      --split "$split" \
      --qa-ids-file "$IDS" \
      --baseline "$BASELINE_ROOT/scores_${split}.jsonl" \
      --out "$out" &
    local pid=$!
    local last="$done" stall=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 20
      local now=0
      [[ -f "$out" ]] && now="$(grep -c . "$out" || true)"
      if [[ "$now" -eq "$last" ]]; then stall=$((stall + 20)); else stall=0; last="$now"; fi
      if [[ "$stall" -ge "$STALL_SECS" ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        break
      fi
    done
    wait "$pid" 2>/dev/null || true
  done
}

for split in valQ valT; do
  score_until_done "$split" "$OUT_ROOT/${split}_dse.jsonl"
  WORK="$WORK" "$PY" archive/implementation/code/analyze_visual_prompt_pilot.py \
    --split "$split" \
    --ids "$IDS" \
    --baseline "$BASELINE_ROOT/scores_${split}.jsonl" \
    --arm dse "$OUT_ROOT/${split}_dse.jsonl" \
    --out "$OUT_ROOT/${split}_report.json"
done
echo "DSE-VTG finished -> $OUT_ROOT"
