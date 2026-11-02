#!/usr/bin/env bash
# Submit Charades grounding on H200, H100, and A100 (separate run dirs).
set -euo pipefail
WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
cd "$WORK"

for spec in h200 h100 a100; do
  job=$(sbatch --parsable "jobs/charades_grounding_train_${spec}.sbatch")
  job=${job%%;*}
  echo "charades_${spec}: $job"
done
