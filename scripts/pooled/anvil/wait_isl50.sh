#!/bin/bash
# Wait until all 40 width-50 tuning-span ensembles exist.
# Waits on the artifact, not on queue emptiness: squeue can read empty in
# the gap between sbatch returning and the scheduler registering the job,
# which made an earlier queue-based wait exit immediately.
while true; do
  n=$(timeout 30 ssh anvil 'ls /anvil/scratch/x-jchang5/results_v2/tune_islands_50/*/ensemble_stitched_predictions.csv 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
  if [ "$n" = "40" ]; then break; fi
  sleep 60
done
echo "ISL50 ALL 40 STITCHED"
