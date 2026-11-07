#!/bin/bash
# Hash-guarded overlay for the live non-git tree. Does not pull, reset, or submit jobs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work)
      WORK=$2
      shift 2
      ;;
    *)
      echo "usage: bash deploy/a100_motion_overlay/deploy.sh --work /scratch/ambpdc/soham/ReVA" >&2
      exit 2
      ;;
  esac
done
WORK=$(cd "$WORK" && pwd)

if grep -F 'runs/b1_q3vl8b_hint16' "$ROOT/MANIFEST_BEFORE.sha256" "$ROOT/MANIFEST_AFTER.sha256" >/dev/null; then
  echo "refusing overlay that names the preserved run directory" >&2
  exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup="$WORK/deploy_backups/motion_overlay_${stamp}"
mkdir -p "$backup"

while read -r hash rel; do
  [[ -z "${hash:-}" || "$hash" == \#* ]] && continue
  if [[ ! -f "$WORK/$rel" ]]; then
    echo "REFUSE missing $WORK/$rel" >&2
    exit 3
  fi
  actual=$(sha256sum "$WORK/$rel" | awk '{print $1}')
  if [[ "$actual" != "$hash" ]]; then
    echo "REFUSE $rel sha256 $actual != expected $hash" >&2
    exit 3
  fi
  mkdir -p "$backup/$(dirname "$rel")"
  cp -a "$WORK/$rel" "$backup/$rel"
done < "$ROOT/MANIFEST_BEFORE.sha256"

while read -r hash rel; do
  [[ -z "${hash:-}" || "$hash" == \#* ]] && continue
  if [[ ! -f "$ROOT/overlay/$rel" ]]; then
    echo "overlay is missing $rel" >&2
    exit 4
  fi
  mkdir -p "$WORK/$(dirname "$rel")"
  cp -a "$ROOT/overlay/$rel" "$WORK/$rel"
  actual=$(sha256sum "$WORK/$rel" | awk '{print $1}')
  if [[ "$actual" != "$hash" ]]; then
    echo "INSTALLED HASH MISMATCH $rel sha256 $actual != $hash" >&2
    exit 4
  fi
done < "$ROOT/MANIFEST_AFTER.sha256"

mkdir -p "$WORK/deploy_backups"
printf '%s\n' "$backup" > "$WORK/deploy_backups/motion_overlay_LATEST"
echo "installed motion overlay"
echo "backup=$backup"
echo "preserved if present: $WORK/runs/b1_q3vl8b_hint16"
echo "next: bash $ROOT/preflight_motion.sh --work $WORK"
