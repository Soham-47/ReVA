#!/bin/bash
# Submit Stage B only after the operator has reviewed Stage A scores.
set -euo pipefail
: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"
for split in valQ valT val; do
  test -s "runs/stageA_q3vl8b_r16/scores_${split}.jsonl" || {
    echo "missing Stage A score file: $split" >&2
    exit 1
  }
  python code/evaluate.py "runs/stageA_q3vl8b_r16/scores_${split}.jsonl" "$split" >/dev/null
done
valq=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["acc"])' runs/stageA_q3vl8b_r16/scores_valQ_summary.json)
valt=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["acc"])' runs/stageA_q3vl8b_r16/scores_valT_summary.json)
val=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["acc"])' runs/stageA_q3vl8b_r16/scores_val_summary.json)
if ! awk -v valq="$valq" -v valt="$valt" -v val="$val" 'BEGIN { exit !(valq >= 0.88 && valt >= 0.78 && val >= 0.82) }'; then
  echo "Stage A gate failed: valQ=$valq valT=$valt val=$val (need 0.88/0.78/0.82)" >&2
  exit 1
fi
stage_b=$(sbatch --parsable --gres=gpu:nvidia_h200_nvl:1 jobs/train.sbatch code/configs/stageB.yaml)
stage_b=${stage_b%%;*}
stage_b_score=$(sbatch --parsable --gres=gpu:nvidia_h200_nvl:1 --dependency="afterok:${stage_b}" jobs/score.sbatch runs/stageB_q3vl8b_r16/final_adapter test val)
stage_b_score=${stage_b_score%%;*}
echo "stage B train: $stage_b"
echo "stage B score: $stage_b_score"
echo "after scoring: python code/make_submission.py --scores runs/stageB_q3vl8b_r16/scores_test.jsonl --out subs/stageB_tta"
