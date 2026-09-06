#!/bin/bash
# Submission for the tuning-span width selection (HP_SCREEN.md, 2026-09-06).
#
#   40 islands: seeds 47-51 only -- seeds 42-46 already exist as
#               probe_TUNE16_ISL40 and are reused.
#   50 islands: seeds 42-51
#   60 islands: seeds 42-51
#
# = 25 array jobs x 4 panels = 100 runs, all three arms at 10 seeds.
#
# --export MUST come BEFORE the script name; putting it after makes sbatch
# treat it as a script argument and every job runs at defaults (the
# 2026-08-27 incident in HP_SCREEN.md).
#
# Usage:  bash islands_tune_submit.sh [--dry-run]

set -u
SBATCH_SCRIPT="$HOME/ONENAS/scripts/pooled/anvil/islands_tune.sbatch"
LOGDIR="/anvil/scratch/x-jchang5/logs_tune_islands"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

mkdir -p "$LOGDIR"
cd "$LOGDIR" || exit 1

n=0
# incumbent: only the seeds the existing tuning fleet lacks
for SEED in 47 48 49 50 51; do
  $DRY sbatch --array=1-4 --export=ALL,ISLANDS=40,SEED="$SEED" "$SBATCH_SCRIPT"
  n=$((n + 1))
done
# candidates: full 10 seeds each
for ISL in 50 60; do
  for SEED in 42 43 44 45 46 47 48 49 50 51; do
    $DRY sbatch --array=1-4 --export=ALL,ISLANDS="$ISL",SEED="$SEED" \
      "$SBATCH_SCRIPT"
    n=$((n + 1))
  done
done
echo "submitted $n array jobs ($((n * 4)) runs)"
