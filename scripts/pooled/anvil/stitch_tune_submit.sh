#!/bin/bash
# Stitch the tuning-span width-selection runs.
#
#   40 islands: seeds 47-51 only.  Seeds 42-46 are the pre-existing
#               probe_TUNE16_ISL40 fleet and are ALREADY stitched.
#   50 islands: seeds 42-51
#   60 islands: seeds 42-51
#
# Skips any run whose ensemble_stitched_predictions.csv already exists, so
# it is safe to re-run after a partial pass (e.g. while one evolution run
# is still finishing).
#
# Usage:  bash stitch_tune_submit.sh [--dry-run]

set -u
SBATCH_SCRIPT="$HOME/ONENAS/scripts/pooled/anvil/stitch_tune.sbatch"
LOGDIR="/anvil/scratch/x-jchang5/logs_stitch_tune"
ROOT="/anvil/scratch/x-jchang5/results_v2"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

mkdir -p "$LOGDIR"
cd "$LOGDIR" || exit 1

submit_arm() {
  local ISL="$1"; shift
  for SEED in "$@"; do
    # only submit if at least one panel still needs stitching, and only
    # if every panel's run finished (300/201-generation elites present)
    local need=0 ready=1
    for p in 1 2 3 4; do
      d="$ROOT/tune_islands_${ISL}/set${p}_seed${SEED}"
      [ -d "$d" ] || { ready=0; continue; }
      ls "$d" 2>/dev/null | grep -q 'generation_20[01]_elites' || ready=0
      [ -f "$d/ensemble_stitched_predictions.csv" ] || need=1
    done
    if [ "$ready" = "0" ]; then
      echo "skip ISLANDS=$ISL SEED=$SEED (evolution not finished)"
      continue
    fi
    if [ "$need" = "0" ]; then
      echo "skip ISLANDS=$ISL SEED=$SEED (already stitched)"
      continue
    fi
    $DRY sbatch --array=1-4 --export=ALL,ISLANDS="$ISL",SEED="$SEED" \
      "$SBATCH_SCRIPT"
    n=$((n + 1))
  done
}

n=0
submit_arm 40 47 48 49 50 51
submit_arm 50 42 43 44 45 46 47 48 49 50 51
submit_arm 60 42 43 44 45 46 47 48 49 50 51
echo "submitted $n array jobs ($((n * 4)) runs)"
