#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash -n jobs/*.sbatch jobs/*.sh
source jobs/wait_child.sh

for file in README.md jobs/train.sbatch jobs/score.sbatch; do
  if ! grep -Fq -- "--gres=gpu:nvidia_h200_nvl:1" "$file"; then
    echo "missing accepted H200 GRES syntax in $file" >&2
    exit 1
  fi
  if grep -Fq -- "--gres=nvidia_h200_nvl:1" "$file"; then
    echo "stale H200 GRES syntax in $file" >&2
    exit 1
  fi
done

for file in jobs/pipeline.sh jobs/submit_stageB.sh; do
  gres_count=$(grep -Fc -- "--gres=gpu:nvidia_h200_nvl:1" "$file" || true)
  if [[ "$gres_count" -ne 2 ]]; then
    echo "each submit helper must pass H200 GRES on both sbatch calls: $file" >&2
    exit 1
  fi
done

for file in jobs/train.sbatch jobs/score.sbatch; do
  if ! grep -Fq -- '#SBATCH -c 12' "$file"; then
    echo "H200 job must request exactly 12 CPUs in $file" >&2
    exit 1
  fi
  if ! grep -Fq 'export HF_HOME="$WORK/hf_cache"' "$file"; then
    echo "missing explicit WORK model cache in $file" >&2
    exit 1
  fi
done

if grep -Eq 'stageA_q3vl8b_r64|stageB_q3vl8b_r64' jobs/*.sh code/configs/*.yaml README.md; then
  echo "stale r64 run path remains" >&2
  exit 1
fi

if ! grep -Fq 'if [[ ! -f "$SMOKE_MARKER" ]]; then' jobs/train.sbatch; then
  echo "train job is missing the smoke marker guard" >&2
  exit 1
fi
if ! grep -Fq 'if [[ -n ${pid:-} ]]; then' jobs/train.sbatch; then
  echo "train job must only create STOP while training is active" >&2
  exit 1
fi
smoke_line=$(grep -nF 'python code/smoke.py' jobs/train.sbatch | cut -d: -f1)
marker_line=$(grep -nF 'touch "$SMOKE_MARKER"' jobs/train.sbatch | cut -d: -f1)
train_line=$(grep -nF 'python code/train.py' jobs/train.sbatch | cut -d: -f1)
if ((smoke_line >= marker_line || marker_line >= train_line)); then
  echo "smoke must pass and write its marker before training" >&2
  exit 1
fi

set +e
(exit 7) & pid=$!
wait_child "$pid"
rc=$?
set -e
test "$rc" -eq 7

(exit 0) & pid=$!
wait_child "$pid"

echo "job wait helper checks passed"
