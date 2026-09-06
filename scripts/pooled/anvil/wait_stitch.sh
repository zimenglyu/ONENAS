#!/bin/bash
# Wait for the stitching array jobs to drain. Interactive helper.
while true; do
  n=$(timeout 30 ssh anvil 'squeue -u $USER -h -r -n onenas_stitch | wc -l' 2>/dev/null)
  if [ -z "$n" ]; then n=99; fi
  if [ "$n" = "0" ]; then break; fi
  sleep 60
done
echo "STITCH QUEUE DRAINED"
