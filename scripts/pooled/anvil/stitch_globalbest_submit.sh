#!/bin/bash
# Single-global-best stitching for the 20- and 60-island eval fleets,
# so the single-vs-ensemble table can report those widths.
# Usage: bash stitch_globalbest_submit.sh [--dry-run]
set -u
S="$HOME/ONENAS/scripts/pooled/anvil/stitch_globalbest.sbatch"
LOG="/anvil/scratch/x-jchang5/logs_globalbest"
OUTROOT="/anvil/scratch/x-jchang5/globalbest"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"
mkdir -p "$LOG"; cd "$LOG" || exit 1
n=0
for TREE in C7_ISL20 islands_60; do
  for SEED in 42 43 44 45 46 47 48 49 50 51; do
    need=0
    for p in 1 2 3 4; do
      [ -f "$OUTROOT/$TREE/set${p}_seed${SEED}/global_best_stitched_predictions.csv" ] || need=1
    done
    if [ "$need" = "0" ]; then
      echo "skip $TREE SEED=$SEED (already done)"; continue
    fi
    $DRY sbatch --array=1-4 --export=ALL,TREE="$TREE",SEED="$SEED" "$S"
    n=$((n + 1))
  done
done
echo "submitted $n array jobs ($((n * 4)) runs)"
