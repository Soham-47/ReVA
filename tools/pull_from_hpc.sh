#!/usr/bin/env bash
# Pull docs/experiments/results (no images/videos) from HPC. Dry-run unless --apply.
# Local files that differ are saved as *.local.bak before being replaced. Never deletes.
set -euo pipefail
HOST="${REVA_HPC_HOST:-ambpdc@hpc.bits-hyderabad.ac.in}"
SRC="${REVA_HPC_DIR:-/scratch/ambpdc/soham/ReVA}"
cd "$(dirname "$0")/.."
MODE=(-n); [[ "${1:-}" == "--apply" ]] && MODE=()
for d in docs experiments results submissions; do
  rsync -rlptc "${MODE[@]}" --itemize-changes --backup --suffix=.local.bak \
    --exclude='__pycache__' --exclude='*.png' --exclude='*.jpg' --exclude='*.mp4' \
    -e "ssh -o BatchMode=yes" "$HOST:$SRC/$d/" "$d/"
done
[[ ${#MODE[@]} -gt 0 ]] && echo "(dry run; pass --apply to copy)"
