#!/bin/bash
# Run this on the Raspberry Pi FIRST, then open the reverse tunnel and submit
# scripts/pooled/anvil/best_run_40isl_pi_global.sbatch or
# scripts/pooled/anvil/best_run_40isl_pi_islands.sbatch (see scripts/pi/README.md).
#
# The pi slices the SAME panel with the SAME flags as the sbatch, so the
# episode ids the master sends for each generation's test window select the
# same data here. Any change to the data flags in the sbatch must be mirrored
# below (the server refuses generations whose episode count does not match).
#
# Usage:  pi_server_40isl.sh <set 1-4> [DATA]
#   DATA     directory holding set{1..4}_core7/*.csv (default ~/panels_core7)
#   PORT     listen port (env, default 5555)
#   INA219=1 enable INA219 power/energy measurement on /dev/i2c-1 (env)
set -euo pipefail
ONENAS="$(cd "$(dirname "$0")/../.." && pwd)"
SET="set${1:-1}"; SET="set${SET#set}"
DATA="${2:-${DATA:-$HOME/panels_core7}}"
PORT="${PORT:-5555}"
PANEL="$DATA/${SET}_core7"
[ -d "$PANEL" ] || { echo "panel dir $PANEL not found"; exit 1; }
FILES=$(ls "$PANEL"/*.csv | grep -v panel_)
INA=""; [ "${INA219:-0}" = "1" ] && INA="--ina219"

OUT="$ONENAS/test_output/pi_server_40isl/$SET"
mkdir -p "$OUT"
echo "pi_genome_server for $SET on port $PORT -> $OUT"
exec "$ONENAS/build/rnn_examples/pi_genome_server" \
  --port "$PORT" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21 \
  --output_parameter_names RET_CS \
  --time_series_length 40 --window_step 5 \
  --normalize none \
  --output_directory "$OUT" \
  --save_genomes $INA \
  --std_message_level INFO --file_message_level INFO
