# Streaming ONE-NAS genomes to a Raspberry Pi

`onenas_mpi --send_to_pi` sends, **once per generation after finalization**,
that generation's genome(s) to a `pi_genome_server` on the pi, together with
the episode ids of the generation's **test window**: the same window whose
predictions the master writes to `generation_<g>_global_best.csv` and
`generation_<g>_elites.csv`. The pi evaluates the genome(s) on exactly that
window, so its numbers are comparable generation by generation with the
master's own files. Without `--send_to_pi` nothing is sent.

Two modes, chosen on the master with `--pi_mode`:

| mode | what is sent each generation | what the pi writes per generation |
|---|---|---|
| `global_best` (default) | the generation's global best genome | `generation_<g>_global_best.csv`, same format as the master's |
| `island_best` | the best genome of every island | `generation_<g>_island_best.csv` (`island,elite_rank,stock,row,predicted`, i.e. the master's elites file restricted to rank 0) and `generation_<g>_ensemble.csv` (mean prediction of the island bests, master's global-best format) |

Every genome gets a row in `pi_evaluations.csv`: generation, mode, island,
genome id, parameter count, MSE / MAE / naive MSE on the window, network build
time, inference time, throughput and (with `--ina219`) INA219 power and energy.
In `island_best` mode the ensemble gets a row with `island = -1, genome_id = -1`.
`--save_genomes` also keeps every received genome under `genomes/`.

## How the pi knows the data

The pi loads and slices **the same files with the same flags** as the master
(`--training_filenames`, `--input_parameter_names`, `--output_parameter_names`,
`--time_offset`, `--time_series_length`, `--window_step`, `--pooled_panel`,
`--normalize`), so episode *i* on the pi is episode *i* on the master; the
master only sends episode ids. Each message carries the master's total episode
count and the pi refuses a generation if its own count differs, which is what
happens when the data flags are out of sync.

Wire format: `common/pi_protocol.hxx`. Sender: `mpi/pi_sender.hxx`
(background thread, retries every 5 s, queue drained before the master exits).
Receiver: `rnn_examples/pi_genome_server.cxx`. Power monitor: `common/ina219.hxx`.

## Anvil -> pi: the 40-island best run on the core7 panels

Two versions of `scripts/pooled/anvil/best_run_40isl.sbatch`, identical to it
apart from the tunnel and the `--send_to_pi` flags:

- `scripts/pooled/anvil/best_run_40isl_pi_global.sbatch` — global best test
- `scripts/pooled/anvil/best_run_40isl_pi_islands.sbatch` — island ensemble test

The pi side is `scripts/pi/pi_server_40isl.sh <set> [DATA]`, which slices the
same panel with the same flags as the sbatch. The pi needs a copy of
`set{1..4}_core7/` (default location `~/panels_core7`). Run one panel-set per
job (`--array=1`, `2`, ...) with the pi serving that set. Through the tunnel
the master talks to `127.0.0.1:5555` (`--pi_host` / `--pi_port` in the sbatch;
the in-code defaults `DEFAULT_PI_HOST` / `DEFAULT_PI_PORT` in `mpi/onenas_mpi.cxx`
only apply when those flags are omitted).

### Laptop -> pi, no tunnel

On the same network the run can also come from a Mac:
```sh
# pi
INA219=1 ./scripts/pi/pi_server_40isl.sh 1 ~/panels_core7
# mac (needs the panels too, DATA=~/panels_core7 by default)
PI_HOST=192.168.0.70 ./scripts/pi/best_run_40isl_pi_mac.sh global_best 1 42
PI_HOST=192.168.0.70 ./scripts/pi/best_run_40isl_pi_mac.sh island_best 1 42
```

### Keys (one time)

| key | make it on | put its `.pub` in |
|---|---|---|
| Anvil key (compute node -> login node) | Anvil | Anvil `~/.ssh/authorized_keys` |
| Pi key (pi -> Anvil) | Pi | Anvil `~/.ssh/authorized_keys` |

On Anvil:
```sh
ssh-keygen -t ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
```
On the pi:
```sh
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub      # append this line to ~/.ssh/authorized_keys on Anvil
```

### Build (one time)

Anvil:
```sh
cd ~/ONENAS && git pull
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0 cmake
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make onenas_mpi
```
Edit the sbatch: `#SBATCH -A`, `ONENAS=`, `OUT=`, `LOGIN_NODE`.

Pi:
```sh
cd ~/Documents/code/ONENAS && git pull
cd build && cmake .. && make pi_genome_server
```

### Run (every time, in this order)

1. Pi, terminal 1 (`INA219=1` to measure power):
   ```sh
   INA219=1 ./scripts/pi/pi_server_40isl.sh 1 ~/panels_core7
   ```
2. Pi, terminal 2 (same `loginNN` as `LOGIN_NODE` in the sbatch):
   ```sh
   ssh -o ServerAliveInterval=60 -R 5555:localhost:5555 x-zlyu2@login03.anvil.rcac.purdue.edu
   ```
3. In that shell:
   ```sh
   cd ~/ONENAS
   sbatch --array=1 --export=ALL,SEED=42 scripts/pooled/anvil/best_run_40isl_pi_global.sbatch
   # or best_run_40isl_pi_islands.sbatch
   ```
4. Keep both pi terminals open until the job finishes. Results land in
   `test_output/pi_server_40isl/set1/` on the pi.
