#!/usr/bin/env bash
# Push code from local git to HPC. Dry-run unless --apply. Never deletes anything on HPC;
# files it would overwrite are saved under .sync_backup/<timestamp>/ on HPC first.
set -euo pipefail
HOST="${REVA_HPC_HOST:-ambpdc@hpc.bits-hyderabad.ac.in}"
DEST="${REVA_HPC_DIR:-/scratch/ambpdc/soham/ReVA}"
cd "$(dirname "$0")/.."
MODE=(-n); [[ "${1:-}" == "--apply" ]] && MODE=()
if [[ ${#MODE[@]} -eq 0 ]] && [[ -n "$(git status --porcelain code jobs tests tools requirements.txt requirements-torch.txt)" ]]; then
  echo "refusing: uncommitted changes in synced paths (commit first)"; exit 1
fi
STAMP=$(date +%Y%m%d_%H%M%S)
rsync -rlptc "${MODE[@]}" --itemize-changes --backup --backup-dir=".sync_backup/$STAMP" \
  --exclude='__pycache__' --exclude='*.pyc' -e "ssh -o BatchMode=yes" \
  code jobs tests tools requirements.txt requirements-torch.txt "$HOST:$DEST/"
[[ ${#MODE[@]} -gt 0 ]] && echo "(dry run; pass --apply to copy)"
