#!/bin/bash
# MacBook version of best_run_40isl_pi_{global,islands}.sbatch: the same
# pinned 40-island run on one core7 panel-set, streaming each generation's
# genome(s) straight to a pi_genome_server on the local network (no SLURM,
# no ssh tunnel). Start scripts/pi/pi_server_40isl.sh <set> on the pi first.
#
# Usage:  best_run_40isl_pi_mac.sh <global_best|island_best> [SET] [SEED] [NP]
#   SET 1..4 (default 1)   SEED default 42   NP MPI ranks, default = cores
#   DATA=~/panels_core7   directory holding set{1..4}_core7/*.csv
#   PI_HOST=192.168.0.70  PI_PORT=5555   the pi's address (env)
#   OUT_ROOT=<repo>/results/best_40isl_pi
set -euo pipefail

MODE="${1:?mode: global_best or island_best}"
case "$MODE" in global_best|island_best) ;; *) echo "mode must be global_best or island_best"; exit 1 ;; esac
ONENAS="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ONENAS/build/mpi/onenas_mpi"
DATA="${DATA:-$HOME/panels_core7}"
OUT_ROOT="${OUT_ROOT:-$ONENAS/results/best_40isl_pi}"
PI_HOST="${PI_HOST:-192.168.0.70}"
PI_PORT="${PI_PORT:-5555}"
SET="set${2:-1}"; SET="set${SET#set}"
SEED="${3:-${SEED:-42}}"
NP="${4:-${NP:-$(sysctl -n hw.ncpu)}}"

command -v mpirun >/dev/null || { echo "mpirun not found: brew install open-mpi"; exit 1; }
[ -x "$BIN" ] || { echo "missing $BIN: build first (cd build && cmake .. && make onenas_mpi)"; exit 1; }
PANEL="$DATA/${SET}_core7"
[ -d "$PANEL" ] || { echo "panel dir $PANEL not found: set DATA=/path/to/panels_core7"; exit 1; }

# eval-span clock (scored span 2020-01-01..2024-12-31), L=40, step 5, V=5
case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

OUT="$OUT_ROOT/${MODE}/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$PANEL"/*.csv | grep -v panel_)

echo "BEST RUN (frozen primary @ 40 islands, $MODE -> pi $PI_HOST:$PI_PORT) $SET seed=$SEED np=$NP -> $OUT"
time mpirun -np "$NP" "$BIN" \
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
