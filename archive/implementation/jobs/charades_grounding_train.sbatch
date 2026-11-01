#!/bin/bash
# Resumable Charades interval SFT/GRPO pilot and paired ReVA evaluation.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_charades
#SBATCH -o reva_charades_%j.out
set -euo pipefail

: "${WORK:=/scratch/ambpdc/soham/ReVA}"
export WORK
source "$WORK/envs/reva/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_CACHE="$WORK/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"
source jobs/wait_child.sh

dataset_root="$WORK/data/Charades-STA/charades_sta"
run_dir="${REVA_RUN_DIR:-$WORK/runs/charades_grounding}"
data_dir="$WORK/runs/charades_grounding/data"
train_rows="$data_dir/train.json"
dev_rows="$data_dir/dev.json"
stage_a="$WORK/runs/stageA_q3vl8b_r16/final_adapter"
job_id="${SLURM_JOB_ID:-manual}"
mkdir -p "$WORK/logs" "$run_dir"
exec > >(tee -a "$WORK/logs/charades_train_${job_id}.log") 2>&1

if [[ ! -f "$data_dir/manifest.json" || ! -f "$train_rows" || ! -f "$dev_rows" ]]; then
  echo "missing prepared Charades data; complete jobs/charades_grounding_data.sbatch first" >&2
  exit 2
fi
if [[ ! -d "$stage_a" ]]; then
  echo "missing frozen Stage A adapter: $stage_a" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/charades_grounding.py preflight --dataset-root "$dataset_root"

graceful=0
pid=""
stop_file=""
forward_signal=0
on_usr1() {
  echo "$(date -Is) received USR1; requesting checkpoint and graceful stop"
  graceful=1
  if ((forward_signal)) && [[ -n "$pid" ]]; then
    kill -USR1 "$pid" 2>/dev/null || true
  fi
  if [[ -n "$stop_file" ]]; then
    touch "$stop_file"
  fi
}
trap on_usr1 USR1

run_step() {
  if ((graceful)); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  "$@" &
  pid=$!
  rc=0
  wait_child "$pid" || rc=$?
  pid=""
  if ((graceful)) || ((rc == 75)); then
    echo "requeueing from the latest safe checkpoint"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if ((rc != 0)); then
    echo "command failed (rc=$rc): $*" >&2
    exit "$rc"
  fi
}

smoke_marker="$run_dir/SMOKE_OK"
if [[ ! -f "$smoke_marker" ]]; then
  run_step python code/charades_grounding.py smoke \
    --dataset-root "$dataset_root" --rows "$train_rows" --init-adapter "$stage_a"
  touch "$smoke_marker"
fi

if [[ ! -f "$run_dir/S0/arm.json" ]]; then
  forward_signal=1
  run_step python code/charades_grounding.py sft --arm s0 \
    --dataset-root "$dataset_root" --train-json "$train_rows" \
    --init-adapter "$stage_a" --output-dir "$run_dir/S0" \
    --max-steps 100 --learning-rate 1e-5 --save-steps 25 --seed 0
  forward_signal=0
fi
if [[ ! -f "$run_dir/S1/arm.json" ]]; then
  forward_signal=1
  run_step python code/charades_grounding.py sft --arm s1 \
    --dataset-root "$dataset_root" --train-json "$train_rows" \
    --init-adapter "$run_dir/S0" --output-dir "$run_dir/S1" \
    --max-steps 100 --learning-rate 1e-5 --save-steps 25 --seed 0
  forward_signal=0
fi
if [[ ! -f "$run_dir/R1/arm.json" ]]; then
  forward_signal=1
  run_step python code/charades_grounding.py grpo --arm r1 \
    --dataset-root "$dataset_root" --train-json "$train_rows" \
    --init-adapter "$run_dir/S0" --reference-adapter "$run_dir/S0" \
    --output-dir "$run_dir/R1" --max-steps 100 --learning-rate 1e-6 \
    --group-size 4 --temperature 0.7 --max-new-tokens 32 \
    --kl-beta 0.02 --clip-range 0.2 --save-steps 25 --seed 0
  forward_signal=0
fi

for arm in A0 S0 S1 R1; do
  adapter="$stage_a"
  [[ "$arm" == A0 ]] || adapter="$run_dir/$arm"
  if [[ ! -f "$run_dir/dev_${arm}/summary.json" ]]; then
    run_step python code/charades_grounding.py evaluate \
      --dataset-root "$dataset_root" --rows "$dev_rows" \
      --adapter "$adapter" --output-dir "$run_dir/dev_${arm}"
  fi
done

for arm in A0 S0 S1 R1; do
  adapter="$stage_a"
  [[ "$arm" == A0 ]] || adapter="$run_dir/$arm"
  for split in valQ valT; do
    score_path="$run_dir/reva_${arm}_${split}.jsonl"
    summary_path="${score_path%.jsonl}_summary.json"
    if [[ ! -f "$summary_path" ]]; then
      stop_file="$run_dir/STOP_${arm}_${split}"
      rm -f "$stop_file"
      export REVA_STOP_FILE="$stop_file"
      run_step python code/score.py --adapter "$adapter" --split "$split" \
        --offsets 0,0.5 --rots 2 --out "$score_path"
      unset REVA_STOP_FILE
      stop_file=""
      run_step python code/evaluate.py "$score_path" "$split"
    fi
  done
done

echo "Charades SFT/GRPO and paired ReVA validation complete: $run_dir"
ls -lh "$run_dir"/dev_*/summary.json "$run_dir"/reva_*_summary.json
