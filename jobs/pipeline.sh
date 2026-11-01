#!/bin/bash
# Submit the selected 8B recipe in dependency order.
set -euo pipefail
: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"

stage_a=$(sbatch --parsable --gres=gpu:nvidia_h200_nvl:1 jobs/train.sbatch code/configs/stageA.yaml)
stage_a=${stage_a%%;*}
stage_a_score=$(sbatch --parsable --gres=gpu:nvidia_h200_nvl:1 --dependency="afterok:${stage_a}" jobs/score.sbatch runs/stageA_q3vl8b_r16/final_adapter valQ valT val)
stage_a_score=${stage_a_score%%;*}
cat <<EOF
stage A train: $stage_a
stage A score: $stage_a_score
EOF
echo "After stage A scoring: for split in valQ valT val; do python code/evaluate.py runs/stageA_q3vl8b_r16/scores_\${split}.jsonl \"\${split}\"; done"
echo "Review Stage A metrics, then submit Stage B explicitly: bash jobs/submit_stageB.sh"
