#!/bin/bash
#SBATCH -J onenas_best40
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#
# THE BEST / FINAL CONFIGURATION — fully pinned, no knobs.
# = the frozen registered primary (PRIMARY.md, ac58a56) at the headline
#   width of 40 islands (Amendment 6 protocol-symmetric selection).
#   Identical to the C7_ISL40 fleet behind the paper's headline tables.
#
# Panels: core7 (RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO
# VOL21 -> RET_CS target), eval clock 2020-2024.
#
# Usage (one panel-set x seed per array task):
#   sbatch --array=1-4 --export=ALL,SEED=42 best_run_40isl.sbatch
#   for s in 42 43 44 45 46 47 48 49 50 51; do
#     sbatch --array=1-4 --export=ALL,SEED=$s best_run_40isl.sbatch; done
#
# ---------------------------------------------------------------------
# THIS RUN DOES NOT ENSEMBLE. It writes, per generation, BOTH
#   generation_N_elites.csv       -- every island's elite population
#   generation_N_global_best.csv  -- the single best genome overall
# and the prediction rule is chosen afterwards, when scoring. The paper's
# rule is the rank-mean ensemble of island champions; the published
# ONE-NAS rule is the single global best. Both come from the same run.
#
# To build the paper's ensemble (NTW must match the value used above for
# this panel set: set1 712, set2 695, set3 702, set4 738):
#
#   module load python/3.9.5
#   python3 $ONENAS/scripts/pooled/score_ensemble.py \
#     --run-dir  "$OUT" \
#     --sidecar-dir "$DATA/${SET}_core7" \
#     --step 5 --length 40 \
#     --num-training-windows "$NTW" --validation-sets 5 \
#     --param RET_CS \
#     --score-from 2020-01-01 \
#     --top-k 10 --book sleeves --hold-days 10 \
#     --ensemble island_champions --combine rank_mean \
#     --emit-json
#
# That writes ensemble_stitched_predictions.csv (date,stock,pred,real,
# naive) next to the run, plus the book and diagnostics. Swap
# --ensemble island_champions for --ensemble global_best to score the
# single-genome rule instead; everything else stays identical, which is
# how the two rules are compared within one run.
# ---------------------------------------------------------------------

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
DATA="/anvil/scratch/x-jchang5/panels_core7"
SEED="${SEED:-42}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

# eval-span clock (scored span 2020-01-01..2024-12-31), L=40, step 5, V=5
case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

OUT="/anvil/scratch/x-jchang5/results_v2/best_40isl/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_core7/"*.csv | grep -v panel_)

echo "BEST RUN (frozen primary @ 40 islands) $SET seed=$SEED -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21 \
  --output_parameter_names RET_CS \
  --number_islands 40 --bp_iterations 10 --num_mutations 1 \
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
