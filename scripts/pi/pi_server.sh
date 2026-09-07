#!/bin/bash
# Run on the Raspberry Pi, terminal 1:   sh scripts/pi/pi_server.sh
# Receives each generation's genome(s) from onenas_mpi and evaluates them on
# that generation's test window of the panel below. Slices the panel with the
# SAME flags as the run scripts; keep them in sync.
# Results: test_output/pi_server/set<SET>/pi_evaluations.csv + per-generation files.

ONENAS="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root

# ---- settings ----
SET=1                          # panel-set 1..4, must match the run script
DATA="$ONENAS/panels_core7"   # holds set{1..4}_core7/*.csv (gitignored)
PORT=5555
INA219=yes                     # yes = measure power on /dev/i2c-1, no = skip
# ------------------

PANEL="$DATA/set${SET}_core7"
[ -d "$PANEL" ] || { echo "panel dir $PANEL not found"; exit 1; }
FILES=$(ls "$PANEL"/*.csv | grep -v panel_)
INA=""; [ "$INA219" = "yes" ] && INA="--ina219"
OUT="$ONENAS/test_output/pi_server/set$SET"
mkdir -p "$OUT"

echo "pi_genome_server: set$SET on port $PORT -> $OUT"
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
