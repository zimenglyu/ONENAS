#!/bin/bash
# Wait for the Anvil islands-sweep queue to drain, then report final states.
# Helper for interactive monitoring; not part of the experiment.
while true; do
  n=$(timeout 30 ssh anvil 'squeue -u $USER -h -r | wc -l' 2>/dev/null)
  if [ -z "$n" ]; then n=999; fi
  if [ "$n" = "0" ]; then break; fi
  sleep 300
done
echo "ISLANDS SWEEP QUEUE DRAINED"
timeout 60 ssh anvil "sacct -S 2026-09-06 --format=State -n | grep -v '\\.' | sort | uniq -c" 2>/dev/null
