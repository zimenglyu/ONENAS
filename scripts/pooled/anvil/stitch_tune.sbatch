#!/bin/bash
#SBATCH -J onenas_stune
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 01:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#
# Build the island-champion rank-mean ensemble for one tuning-span run
# (HP_SCREEN.md width selection, declared 2026-09-06).  Same as
# stitch_islands.sbatch but against the tune16 clock and the
# tune_islands_* output tree.
#
# One array task per panel; ISLANDS and SEED select the arm and seed, so
# the submission pattern matches islands_tune.sbatch exactly.
#
# Usage (--export BEFORE the script name):
#   sbatch --array=1-4 --export=ALL,ISLANDS=50,SEED=42 stitch_tune.sbatch

module load python/3.9.5

if [ -z "$ISLANDS" ]; then echo "FATAL: ISLANDS not set"; exit 1; fi
if [ -z "$SEED" ];    then echo "FATAL: SEED not set";    exit 1; fi

SET="set${SLURM_ARRAY_TASK_ID:-1}"

# tune16 clock, matching islands_tune.sbatch
case "$SET" in
  set1) NTW=511 ;;
  set2) NTW=493 ;;
  set3) NTW=501 ;;
  set4) NTW=536 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

RUN="/anvil/scratch/x-jchang5/results_v2/tune_islands_${ISLANDS}/${SET}_seed${SEED}"
PAN="/anvil/scratch/x-jchang5/panels_core7/${SET}_core7"

if [ ! -d "$RUN" ]; then echo "FATAL: no run dir $RUN"; exit 1; fi

echo "stitching TUNE islands=$ISLANDS $SET seed=$SEED"
python3 "$HOME/ONENAS/scripts/pooled/score_ensemble.py" \
  --run-dir "$RUN" \
  --sidecar-dir "$PAN" \
  --step 5 --length 40 \
  --num-training-windows "$NTW" --validation-sets 5 \
  --param RET_CS \
  --score-from 2016-01-01 \
  --top-k 10 --book sleeves --hold-days 10 \
  --ensemble island_champions --combine rank_mean \
  --emit-json
