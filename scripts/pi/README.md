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

## The scripts

Every script runs with no arguments; its settings (panel-set, seed, addresses)
are the variables at the top of the file. Set `SET` to the same panel-set in
the pi server and in the run script.

| where | script | what |
|---|---|---|
| pi | `scripts/pi/pi_server.sh` | evaluates what the master sends (settings: SET, DATA, PORT, INA219) |
| pi | `scripts/pi/pi_tunnel.sh` | reverse tunnel to Anvil for the cluster runs (settings: ANVIL_USER, LOGIN_NODE, PORT) |
| Anvil | `scripts/pooled/anvil/best_run_40isl_pi_global.sbatch` | the 40-island best run, global best test (settings: SET, SEED, LOGIN_NODE) |
| Anvil | `scripts/pooled/anvil/best_run_40isl_pi_islands.sbatch` | the 40-island best run, island ensemble test (settings: SET, SEED, LOGIN_NODE) |
| Mac | `scripts/pi/mac_global.sh` | same run from a laptop on the pi's network, global best test (settings: SET, SEED, DATA, PI_HOST) |
| Mac | `scripts/pi/mac_islands.sh` | same run from a laptop, island ensemble test |

The two sbatch files and the two Mac scripts are `best_run_40isl.sbatch` plus
the pi flags; the ONE-NAS flags are identical. The pi needs a copy of the
panels, and so does the Mac: unzip `panels_core7.zip` in the repo root so that
`panels_core7/set{1..4}_core7/*.csv` exists (the directory is gitignored).

## Setup (once)

Pi:
```sh
cd ~/Documents/code/ONENAS && git pull
cd build && cmake .. && make pi_genome_server
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub      # append this line to ~/.ssh/authorized_keys on Anvil
```

Anvil:
```sh
cd ~/ONENAS && git pull
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0 cmake
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make onenas_mpi
ssh-keygen -t ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
```
(the compute node reaches the login node with the Anvil key; the pi reaches
Anvil with the pi key.) Set `#SBATCH -A`, `ONENAS=` and `OUT=` in the sbatch
files for your account.

## Run: Anvil -> pi

1. Pi, terminal 1:
   ```sh
   sh scripts/pi/pi_server.sh
   ```
2. Pi, terminal 2 (opens the tunnel and leaves you on the login node):
   ```sh
   sh scripts/pi/pi_tunnel.sh
   ```
3. In that login-node shell, one of:
   ```sh
   cd ~/ONENAS
   sbatch scripts/pooled/anvil/best_run_40isl_pi_global.sbatch
   sbatch scripts/pooled/anvil/best_run_40isl_pi_islands.sbatch
   ```
4. Keep both pi terminals open until the job finishes.

## Run: Mac -> pi

1. Pi:
   ```sh
   sh scripts/pi/pi_server.sh
   ```
2. Mac, one of:
   ```sh
   sh scripts/pi/mac_global.sh
   sh scripts/pi/mac_islands.sh
   ```

Results are on the pi in `test_output/pi_server/set<SET>/`: `pi_evaluations.csv`,
the per-generation prediction files and the received genomes under `genomes/`.
