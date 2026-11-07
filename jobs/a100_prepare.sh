#!/bin/bash
# Shared setup for the A100 specialist wrappers. Source this file; do not sbatch it.

a100_prepare_env() {
  : "${WORK:?set WORK to the runtime Lustre tree}"
  if [[ -f "$WORK/jobs/activate_reva.sh" ]]; then
    # shellcheck disable=SC1091
    source "$WORK/jobs/activate_reva.sh"
  elif [[ -f "$WORK/venv312/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$WORK/venv312/bin/activate"
  else
    echo "refusing to guess an environment; expected $WORK/jobs/activate_reva.sh or $WORK/venv312" >&2
    return 2
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
  export PYTHONUNBUFFERED=1
  # Match the live jobs/train.sbatch allocator. Do not use the expandable-segments setting.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
  export TORCH_CUDNN_V8_API_DISABLED=1
  export CUDNN_FRONTEND_DISABLE=1
  cd "$WORK"
}

a100_resolve_output() {
  local cfg="$1"
  local run_name root out stamp jobtag resolved
  run_name=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_name"])' "$cfg")
  root="$WORK/runs/specialists/$run_name"
  mkdir -p "$root"
  out=""
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local matches=()
    local nullglob_was=0
    if shopt -q nullglob; then
      nullglob_was=1
    fi
    shopt -s nullglob
    matches=("$root"/*_"${SLURM_JOB_ID}")
    if ((nullglob_was)); then
      shopt -s nullglob
    else
      shopt -u nullglob
    fi
    if ((${#matches[@]} > 1)); then
      echo "multiple output directories for job ${SLURM_JOB_ID}" >&2
      return 2
    elif ((${#matches[@]} == 1)); then
      out="${matches[0]}"
    fi
  fi
  if [[ -z "$out" ]]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    jobtag="${SLURM_JOB_ID:-noslurm}"
    out="$root/${stamp}_${jobtag}"
    if [[ -e "$out" ]]; then
      echo "refusing to overwrite $out" >&2
      return 2
    fi
    mkdir -p "$out"
  fi
  resolved="$out/resolved_config.yaml"
  if [[ ! -f "$resolved" ]]; then
    python code/specialist_data.py materialize-config "$cfg" "$resolved" "$out" --work "$WORK" >/dev/null
  fi
  printf '%s\n' "$resolved"
}
