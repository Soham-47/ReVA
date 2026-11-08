#!/bin/bash
# Source from Slurm job scripts after setting WORK.
set -euo pipefail
: "${WORK:?set WORK to the ReVA runtime tree}"

if [[ -f "$WORK/venv312/bin/activate" ]]; then
  source "$WORK/venv312/bin/activate"
elif [[ -f "$WORK/envs/reva/bin/activate" ]]; then
  source "$WORK/envs/reva/bin/activate"
else
  echo "ReVA venv missing: expected $WORK/venv312/bin/activate or $WORK/envs/reva/bin/activate" >&2
  exit 1
fi
