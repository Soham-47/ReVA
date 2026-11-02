#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J reva_motion_cache
#SBATCH -o reva_motion_cache_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
source "$WORK/envs/reva/bin/activate"
cd "$WORK"

echo "Building camera motion cache on $(hostname) at $(date)..."
python archive/implementation/code/build_camera_motion_cache.py --out data/camera_motion_hints.json --splits fit_A,valQ,valT
echo "Camera motion cache complete!"
