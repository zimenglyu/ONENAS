#!/bin/bash
#SBATCH -J onenas_stitch
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 01:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#
# Build the island-champion rank-mean ensemble for one run of the width
# sweep.  Stitching is single-core pure Python, so it runs in the shared
# partition rather than burning a whole node; one array task per run.
#
# Array index -> (panel, seed):  id 1..40, panel = ((id-1) % 4) + 1,
# seed = 42 + (id-1)/4.  So ids 1-4 are set1-4 at seed 42, and so on.
#
# Usage (--export BEFORE the script name, as always):
#   sbatch --array=1-40 --export=ALL,ISLANDS=30 stitch_islands.sbatch

module load python/3.9.5

if [ -z "$ISLANDS" ]; then echo "FATAL: ISLANDS not set"; exit 1; fi

ID=${SLURM_ARRAY_TASK_ID:-1}
PANEL=$(( (ID - 1) % 4 + 1 ))
SEED=$(( 42 + (ID - 1) / 4 ))
SET="set${PANEL}"

case "$SET" in
  set1) NTW=712 ;;
  set2) NTW=695 ;;
  set3) NTW=702 ;;
  set4) NTW=738 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

RUN="/anvil/scratch/x-jchang5/results_v2/islands_${ISLANDS}/${SET}_seed${SEED}"
PAN="/anvil/scratch/x-jchang5/panels_core7/${SET}_core7"

if [ ! -d "$RUN" ]; then echo "FATAL: no run dir $RUN"; exit 1; fi

echo "stitching islands=$ISLANDS $SET seed=$SEED"
python3 "$HOME/ONENAS/scripts/pooled/score_ensemble.py" \
  --run-dir "$RUN" \
  --sidecar-dir "$PAN" \
  --step 5 --length 40 \
  --num-training-windows "$NTW" --validation-sets 5 \
  --param RET_CS \
  --score-from 2020-01-01 \
  --top-k 10 --book sleeves --hold-days 10 \
  --ensemble island_champions --combine rank_mean \
  --emit-json
