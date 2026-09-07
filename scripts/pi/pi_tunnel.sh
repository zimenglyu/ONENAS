#!/bin/bash
# Run on the Raspberry Pi, terminal 2:   sh scripts/pi/pi_tunnel.sh
# Opens the reverse tunnel so an Anvil job can reach pi_server.sh, and leaves
# you in a shell on the login node: submit the sbatch from THAT shell and keep
# this terminal open until the job finishes.

# ---- settings ----
ANVIL_USER=x-zlyu2
LOGIN_NODE=login03             # must match LOGIN_NODE in the sbatch
PORT=5555
# ------------------

exec ssh -o ServerAliveInterval=60 -R "$PORT:localhost:$PORT" "$ANVIL_USER@$LOGIN_NODE.anvil.rcac.purdue.edu"
