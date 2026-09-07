#!/bin/bash
# Run on a Mac on the pi's network:   sh scripts/pi/mac_global.sh
# The pinned 40-island best run on one panel-set, streaming the generation's GLOBAL BEST genome
# to pi_server.sh after every generation. Start pi_server.sh on the pi first.
# Same ONE-NAS flags as scripts/pooled/anvil/best_run_40isl.sbatch.

ONENAS="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root

# ---- settings ----
SET=1                          # panel-set 1..4, must match pi_server.sh
SEED=42
DATA="$ONENAS/panels_core7"   # holds set{1..4}_core7/*.csv (gitignored)
PI_HOST=192.168.0.70
PI_PORT=5555
NP=$(sysctl -n hw.ncpu)        # MPI ranks
# ------------------

MODE=global_best
PANEL="$DATA/set${SET}_core7"
[ -d "$PANEL" ] || { echo "panel dir $PANEL not found"; exit 1; }
FILES=$(ls "$PANEL"/*.csv | grep -v panel_)
case "$SET" in   # eval-span clock (scored span 2020-01-01..2024-12-31), L=40, step 5, V=5
  1) NTW=712; TOTGEN=301 ;;
  2) NTW=695; TOTGEN=300 ;;
  3) NTW=702; TOTGEN=300 ;;
  4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac
OUT="$ONENAS/results/best_40isl_pi/$MODE/set${SET}_seed${SEED}"
mkdir -p "$OUT"

echo "BEST RUN (40 islands, $MODE -> pi $PI_HOST:$PI_PORT) set$SET seed=$SEED -> $OUT"
time mpirun -np "$NP" "$ONENAS/build/mpi/onenas_mpi" \
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
  --send_to_pi --pi_mode "$MODE" --pi_host "$PI_HOST" --pi_port "$PI_PORT" \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUT" 2>&1 | tee "$OUT/run.log"
