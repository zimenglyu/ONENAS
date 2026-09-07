#!/bin/bash
#SBATCH -J onenas_sgb
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 01:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#
# Single-global-best predictions for one eval-span run, for the
# single-vs-ensemble table. Same run, same clock, same book as the
# ensemble stitch -- only --ensemble global_best differs, which is what
# makes the comparison within-run and paired.
#
# TREE selects the fleet directory (its runs must be eval-clock):
#   C7_ISL20 / C7_ISL40 / islands_60 / ...
# Output goes to global_best_stitched_predictions.csv beside the run, so
# it never overwrites the ensemble file.
#
# Usage (--export BEFORE the script name):
#   sbatch --array=1-4 --export=ALL,TREE=C7_ISL20,SEED=42 stitch_globalbest.sbatch

module load python/3.9.5

if [ -z "$TREE" ]; then echo "FATAL: TREE not set"; exit 1; fi
if [ -z "$SEED" ]; then echo "FATAL: SEED not set"; exit 1; fi

SET="set${SLURM_ARRAY_TASK_ID:-1}"

# eval-span clock, matching the runs in these trees
case "$SET" in
  set1) NTW=712 ;;
  set2) NTW=695 ;;
  set3) NTW=702 ;;
  set4) NTW=738 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

RUN="/anvil/scratch/x-jchang5/results_v2/${TREE}/${SET}_seed${SEED}"
PAN="/anvil/scratch/x-jchang5/panels_core7/${SET}_core7"
OUT="/anvil/scratch/x-jchang5/globalbest/${TREE}/${SET}_seed${SEED}"

if [ ! -d "$RUN" ]; then echo "FATAL: no run dir $RUN"; exit 1; fi
mkdir -p "$OUT"

echo "global_best stitch $TREE $SET seed=$SEED"
python3 "$HOME/ONENAS/scripts/pooled/score_ensemble.py" \
  --run-dir "$RUN" \
  --sidecar-dir "$PAN" \
  --out-dir "$OUT" \
  --step 5 --length 40 \
  --num-training-windows "$NTW" --validation-sets 5 \
  --param RET_CS \
  --score-from 2020-01-01 \
  --top-k 10 --book sleeves --hold-days 10 \
  --ensemble global_best \
  --emit-json
