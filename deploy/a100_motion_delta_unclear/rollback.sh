#!/bin/bash
# Restore the hash-checked backup created by deploy.sh. Does not submit or cancel jobs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
BACKUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work)
      WORK=$2
      shift 2
      ;;
    --backup)
      BACKUP=$2
      shift 2
      ;;
    *)
      echo "usage: bash deploy/a100_motion_overlay/rollback.sh --work WORK --backup BACKUP_DIR" >&2
      exit 2
      ;;
  esac
done
if [[ -z "$BACKUP" && -f "$WORK/deploy_backups/motion_overlay_LATEST" ]]; then
  BACKUP=$(cat "$WORK/deploy_backups/motion_overlay_LATEST")
fi
if [[ -z "$BACKUP" || ! -d "$BACKUP" ]]; then
  echo "backup directory is missing" >&2
  exit 2
fi
WORK=$(cd "$WORK" && pwd)

while read -r hash rel; do
  [[ -z "${hash:-}" || "$hash" == \#* ]] && continue
  if [[ ! -f "$WORK/$rel" ]]; then
    echo "REFUSE rollback; missing installed $rel" >&2
    exit 3
  fi
  actual=$(sha256sum "$WORK/$rel" | awk '{print $1}')
  if [[ "$actual" != "$hash" ]]; then
    echo "REFUSE rollback; $rel was changed after deploy (sha256 $actual != $hash)" >&2
    exit 3
  fi
done < "$ROOT/MANIFEST_AFTER.sha256"

while read -r hash rel; do
  [[ -z "${hash:-}" || "$hash" == \#* ]] && continue
  if [[ ! -f "$BACKUP/$rel" ]]; then
    echo "backup is missing $rel" >&2
    exit 4
  fi
  cp -a "$BACKUP/$rel" "$WORK/$rel"
  actual=$(sha256sum "$WORK/$rel" | awk '{print $1}')
  if [[ "$actual" != "$hash" ]]; then
    echo "ROLLBACK HASH MISMATCH $rel sha256 $actual != $hash" >&2
    exit 4
  fi
done < "$ROOT/MANIFEST_BEFORE.sha256"

if [[ -f "$ROOT/NEW_FILES.txt" ]]; then
  while read -r rel; do
    [[ -z "${rel:-}" || "$rel" == \#* ]] && continue
    rm -f "$WORK/$rel"
  done < "$ROOT/NEW_FILES.txt"
fi
echo "rolled back to $BACKUP"
