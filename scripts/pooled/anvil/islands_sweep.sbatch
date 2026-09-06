#!/bin/bash
#SBATCH -J onenas_isl
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 03:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#
# WIDTH-CURVE REFRESH (HP_SCREEN.md, declared 2026-09-06).
# The frozen registered primary in every respect EXCEPT --number_islands,
# on the eval clock (2020-2024). Descriptive figure only -- see the
# declaration: no width here is headline-eligible without the 2016-2019
# protocol-symmetric selection plus a PRIMARY.md amendment.
#
# ntasks is pinned at 128 for every width so the worker pool is identical
# across the curve; only the island count varies.
#
# Usage (ISLANDS and SEED are REQUIRED; --export MUST precede the script
# name -- see the 2026-08-27 incident in HP_SCREEN.md):
#   sbatch --array=1-4 --export=ALL,ISLANDS=30,SEED=42 islands_sweep.sbatch
#
# Widths 20 and 40 are deliberately NOT part of this sweep; the existing
# n=10 eval-span fleets at those widths are reused.

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
DATA="/anvil/scratch/x-jchang5/panels_core7"

if [ -z "$ISLANDS" ]; then echo "FATAL: ISLANDS not set"; exit 1; fi
if [ -z "$SEED" ];    then echo "FATAL: SEED not set";    exit 1; fi

SET="set${SLURM_ARRAY_TASK_ID:-1}"

# eval-span clock (scored span 2020-01-01..2024-12-31), L=40, step 5, V=5
case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

OUT="/anvil/scratch/x-jchang5/results_v2/islands_${ISLANDS}/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_core7/"*.csv | grep -v panel_)

echo "WIDTH SWEEP islands=$ISLANDS $SET seed=$SEED -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21 \
  --output_parameter_names RET_CS \
  --number_islands "$ISLANDS" --bp_iterations 10 --num_mutations 1 \
  --time_series_length 40 --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets 5 --num_training_sets 2000 \
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation 1 \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --selection_metric mse \
  --max_pred_sd_ratio 3.0 \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --normalize none --compare_with_naive --control_size_method none \
  --write_elite_predictions \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUT"
