#!/bin/bash
#SBATCH -J onenas_best40_pi_global
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 19:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#
# THE BEST / FINAL CONFIGURATION, streamed to a Raspberry Pi (global_best).
# Same run as best_run_40isl.sbatch; additionally, after every generation,
# the generation's GLOBAL BEST genome
# is sent to a pi_genome_server, which evaluates it on that generation's test
# window (see scripts/pi/README.md). The pi must be running
#   scripts/pi/pi_server.sh       (with the same SET as below)
# and holding open a reverse tunnel to LOGIN_NODE (below) before this is submitted.
# One panel-set per job: the pi evaluates one panel-set at a time.
#
# = the frozen registered primary (PRIMARY.md, ac58a56) at the headline
#   width of 40 islands (Amendment 6 protocol-symmetric selection).
#   Identical to the C7_ISL40 fleet behind the paper's headline tables.
#
# Panels: core7 (RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO
# VOL21 -> RET_CS target), eval clock 2020-2024, sleeves-book scoring via
# score_ensemble.py --ensemble island_champions --combine rank_mean.
#
# Run from the login-node shell opened by scripts/pi/pi_tunnel.sh:
#   sbatch scripts/pooled/anvil/best_run_40isl_pi_global.sh
# Settings (SET, SEEDS, LOGIN_NODE) are the variables below. The seeds run one
# after another inside this one job, because the pi serves one run at a time --
# do NOT submit several of these at once, they would all reach for the same pi.

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/code/ONENAS"
DATA="/anvil/projects/x-cis251123/shared/panels_core7"

# ---- settings ----
SET=1                  # panel-set 1..4, must match pi_server.sh on the pi
SEEDS="42 43 44 45 46 47 48 49 50 51"   # run sequentially, one result set each
LOGIN_NODE="login03"   # the login node pi_tunnel.sh is attached to
PI_PORT=5555
# ------------------
SET="set$SET"

# forward this compute node's $PI_PORT to the login node, where the pi's
# reverse tunnel is listening (see scripts/pi/README.md for the keys)
ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -L $PI_PORT:localhost:$PI_PORT $LOGIN_NODE &
TUNNEL_PID=$!
sleep 3
if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "could not open tunnel to $LOGIN_NODE, genomes will not reach the pi"
fi

# eval-span clock (scored span 2020-01-01..2024-12-31), L=40, step 5, V=5
case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

FILES=$(ls "$DATA/${SET}_core7/"*.csv | grep -v panel_)

RUN=0
for SEED in $SEEDS; do
RUN=$((RUN+1))
OUT="/anvil/scratch/x-zlyu2/results_v2/best_40isl_pi_global/${SET}_seed${SEED}"
mkdir -p "$OUT"
echo "### run $RUN of $(echo $SEEDS | wc -w): seed $SEED  ($(date))"
  echo "BEST RUN (frozen primary @ 40 islands, global_best -> pi) $SET seed=$SEED -> $OUT"
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
    --send_to_pi --pi_mode global_best --pi_host 127.0.0.1 --pi_port $PI_PORT \
    --std_message_level INFO --file_message_level INFO \
    --output_directory "$OUT"

done

kill $TUNNEL_PID 2>/dev/null
