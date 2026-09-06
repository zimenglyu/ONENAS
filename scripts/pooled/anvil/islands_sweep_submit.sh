#!/bin/bash
# Mass submission for the width-curve refresh (HP_SCREEN.md, 2026-09-06).
#
# Widths 10/30/50/60 x panels set1-4 x seeds 42-51.  Widths 20 and 40 are
# NOT here: the existing n=10 eval-span fleets at those widths are the same
# configuration on the same clock and are reused.
#
# --export MUST come BEFORE the script name.  Putting it after makes sbatch
# treat it as a script argument, every job silently runs at defaults, and
# the output clobbers whatever directory the defaults point at -- that is
# the 2026-08-27 incident recorded in HP_SCREEN.md.  Do not reorder.
#
# Already submitted by hand as the smoke pair (do not resubmit):
#   ISLANDS=10 SEED=42   (job 20432783, COMPLETED, stitch verified)
#   ISLANDS=60 SEED=42   (job 20432788)
#
# Usage:  bash islands_sweep_submit.sh [--dry-run]

set -u
SBATCH_SCRIPT="$HOME/ONENAS/scripts/pooled/anvil/islands_sweep.sbatch"
LOGDIR="/anvil/scratch/x-jchang5/logs_islands"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

mkdir -p "$LOGDIR"
cd "$LOGDIR" || exit 1

n=0
for ISL in 10 30 50 60; do
  for SEED in 42 43 44 45 46 47 48 49 50 51; do
    # skip the two cells already submitted as the smoke pair
    if { [ "$ISL" = "10" ] || [ "$ISL" = "60" ]; } && [ "$SEED" = "42" ]; then
      echo "skip ISLANDS=$ISL SEED=$SEED (smoke pair, already submitted)"
      continue
    fi
    $DRY sbatch --array=1-4 --export=ALL,ISLANDS="$ISL",SEED="$SEED" \
      "$SBATCH_SCRIPT"
    n=$((n + 1))
  done
done
echo "submitted $n array jobs ($((n * 4)) runs)"
