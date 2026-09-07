// Runs on the Raspberry Pi. Listens on a TCP port for the per-generation
// messages streamed by onenas_mpi (--send_to_pi) and evaluates each
// generation's genome(s) on that generation's TEST WINDOW: the same window
// whose predictions the master writes to generation_<g>_global_best.csv and
// generation_<g>_elites.csv.
//
// The pi needs the same data as the master, sliced the same way. Give it the
// same --training_filenames, --input_parameter_names, --output_parameter_names,
// --time_offset, --time_series_length, --window_step, --pooled_panel and
// --normalize flags as onenas_mpi: episode i on the pi is then identical to
// episode i on the master, and every message names the episode ids of its
// test window (one per stock in pooled panel mode). The pi checks that its
// episode count matches the master's and refuses a generation otherwise.
//
// Modes (chosen on the master with --pi_mode):
//   global_best: one genome per generation, the generation's global best.
//                Writes generation_<g>_global_best.csv in the master's format.
//   island_best: the best genome of every island. Each is scored on its own
//                and their mean prediction is scored as the ensemble. Writes
//                generation_<g>_island_best.csv (island,elite_rank,stock,row,
//                predicted, like the master's elites file restricted to rank 0)
//                and generation_<g>_ensemble.csv (master's global-best format).
//
// Each run (one --pi_mode and one --online_series_seed) gets its own directory,
// <output_directory>/<mode>_seed<seed>/, so back-to-back runs never overwrite one
// another and the pi does not have to be restarted between them. The idle
// baseline is re-metered at the start of every run, while the pi is genuinely
// idle waiting on the connection.
//
// Every genome gets a row in pi_evaluations.csv with MSE/MAE on the window,
// the naive (previous value) MSE, inference time, throughput and, with
// --ina219, INA219 power/energy during inference. One inference over the
// window takes milliseconds, too short to meter, so the inference is repeated
// until at least --min_measure_ms (default 100) have elapsed and the time and
// energy are reported PER INFERENCE (total / repeats). The board's idle power
// is metered for --idle_measure_ms (default 2000) at startup and reported as
// idle_power_mw; energy_net_mj is the energy above idle. The ensemble row has
// island = -1 and genome_id = -1.

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <fstream>
using std::ofstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/ina219.hxx"
#include "common/log.hxx"
#include "common/pi_protocol.hxx"
#include "common/process_arguments.hxx"
#include "rnn/rnn.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;

typedef vector<vector<vector<double> > > Series3;  // [series][parameter][timestep]

// reads exactly length bytes, returns false when the connection is closed
bool read_all(int fd, char* buffer, int32_t length) {
    int32_t total = 0;
    while (total < length) {
        ssize_t n = recv(fd, buffer + total, length - total, 0);
        if (n <= 0) {
            return false;
        }
        total += n;
    }
    return true;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
}

// MSE / MAE / naive MSE over timesteps 1..T-1 of every series and output, exactly
// as OneNasIslandSpeciationStrategy::calculate_prediction_performance scores the master's file
void score(const Series3& predictions, const Series3& expected, double& mse, double& mae, double& naive_mse, int32_t& rows) {
    mse = mae = naive_mse = 0.0;
    rows = 0;
    for (int32_t n = 0; n < (int32_t) expected.size(); n++) {
        for (int32_t i = 0; i < (int32_t) expected[n].size(); i++) {
            for (int32_t j = 1; j < (int32_t) expected[n][i].size(); j++) {
                double e = expected[n][i][j] - predictions[n][i][j];
                double ne = expected[n][i][j] - expected[n][i][j - 1];
                mse += e * e;
                mae += fabs(e);
                naive_mse += ne * ne;
                rows++;
            }
        }
    }
    if (rows > 0) {
        mse /= rows;
        mae /= rows;
        naive_mse /= rows;
    }
}

// the master's generation_<g>_global_best.csv layout, with a configurable prediction column prefix
void write_global_format(const string& filename, const string& prefix, const vector<string>& output_names, const Series3& predictions, const Series3& expected) {
    ofstream out(filename);
    int32_t num_series = (int32_t) expected.size();
    int32_t num_outputs = (int32_t) output_names.size();
    out << "#";
    for (int32_t n = 0; n < num_series; n++) {
        string suffix = (num_series > 1) ? ("_s" + std::to_string(n)) : "";
        for (int32_t i = 0; i < num_outputs; i++) out << (n > 0 || i > 0 ? "," : "") << "expected_" << output_names[i] << suffix;
        for (int32_t i = 0; i < num_outputs; i++) out << ",naive_" << output_names[i] << suffix;
        for (int32_t i = 0; i < num_outputs; i++) out << "," << prefix << output_names[i] << suffix;
    }
    out << "\n";
    int32_t time_length = (int32_t) expected[0][0].size();
    for (int32_t j = 1; j < time_length; j++) {
        for (int32_t n = 0; n < num_series; n++) {
            for (int32_t i = 0; i < num_outputs; i++) out << (n > 0 || i > 0 ? "," : "") << expected[n][i][j];
            for (int32_t i = 0; i < num_outputs; i++) out << "," << expected[n][i][j - 1];
            for (int32_t i = 0; i < num_outputs; i++) out << "," << predictions[n][i][j];
        }
        out << "\n";
    }
}

struct GenomeResult {
    int32_t island, genome_id, parameters;
    double mse, mae, naive_mse, build_ms;
    double inference_ms;   // per inference over the window (measured_ms / repeats)
    int32_t repeats;       // inferences run inside the measured window
    double measured_ms;    // length of the metered window
    int32_t rows;
    INA219Stats power;     // over the metered window; energy_mj is per inference
    double energy_net_mj;  // per inference, above the idle baseline
    Series3 predictions;
};

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t port;
    get_argument(arguments, "--port", true, port);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);
    mkdir(output_directory.c_str(), 0755);

    bool save_genomes = argument_exists(arguments, "--save_genomes");
    bool write_predictions = !argument_exists(arguments, "--no_prediction_files");

    bool use_ina219 = argument_exists(arguments, "--ina219");
    string ina219_device = "/dev/i2c-1";
    get_argument(arguments, "--ina219_device", false, ina219_device);
    double min_measure_ms = 100.0;
    get_argument(arguments, "--min_measure_ms", false, min_measure_ms);
    int32_t idle_measure_ms = 2000;
    get_argument(arguments, "--idle_measure_ms", false, idle_measure_ms);

    // the same data, sliced the same way as onenas_mpi's main()
    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    Series3 inputs, outputs;
    slice_online_time_series(arguments, time_series_sets, inputs, outputs);
    vector<string> output_names = time_series_sets->get_output_parameter_names();
    Log::info("sliced %d episodes (%d inputs x %d timesteps each)\n", (int32_t) inputs.size(), (int32_t) inputs[0].size(), (int32_t) inputs[0][0].size());

    INA219 ina219;
    INA219Sampler ina219_sampler;
    bool ina219_active = false;
    double idle_power_mw = 0.0;   // re-metered at the start of every run
    if (use_ina219) {
        if (ina219.open_device(ina219_device.c_str()) && ina219.configure()) {
            INA219Reading first;
            if (!ina219.read_reading(first)) {
                Log::warning("INA219 found on %s but a reading failed, continuing without power monitoring\n", ina219_device.c_str());
            } else {
                ina219_active = true;
                Log::info("INA219 power monitor enabled on %s: bus %.3f V, shunt %.3f mV, current %.1f mA, power %.1f mW\n",
                    ina219_device.c_str(), first.bus_voltage_v, first.shunt_voltage_mv, first.current_ma, first.power_mw);
                if (first.bus_voltage_v < 0.5) {
                    Log::warning("INA219 bus voltage is ~0 V: VIN- is not connected to the load, check the wiring\n");
                }
                if (fabs(first.current_ma) < 1.0) {
                    Log::warning("INA219 current is ~0 mA: the shunt is not in the supply path, check the wiring\n");
                }
            }
        } else {
            Log::warning("INA219 requested but could not open %s, continuing without power monitoring\n", ina219_device.c_str());
        }
    }

    // per-run output state; a run is one (mode, seed) pair
    string run_key, results_directory;
    ofstream results;
    Log::info("results will go to %s/<mode>_seed<seed>/ (pi_evaluations.csv, per-generation prediction files%s)\n",
        output_directory.c_str(), save_genomes ? ", genomes/" : "");

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(listen_fd, (struct sockaddr*) &addr, sizeof(addr)) < 0 || listen(listen_fd, 1) < 0) {
        Log::fatal("could not listen on port %d\n", port);
        return 1;
    }

    while (true) {
        Log::info("waiting for a connection on port %d\n", port);
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            continue;
        }
        Log::info("connected\n");

        int32_t length;
        while (read_all(fd, (char*) &length, sizeof(int32_t))) {
            vector<char> bytes(length);
            if (!read_all(fd, bytes.data(), length)) {
                break;
            }
            PiGenerationMessage msg;
            string error;
            if (!msg.parse(bytes.data(), length, error)) {
                Log::error("dropping a %d byte message: %s\n", length, error.c_str());
                continue;
            }
            int32_t g = msg.generation;
            Log::info("generation %d: %d genome(s), mode %s, seed %d, %d test episode(s)\n", g, (int32_t) msg.genome_bytes.size(), pi_mode_name(msg.mode).c_str(), msg.seed, (int32_t) msg.test_episode_ids.size());

            // a new (mode, seed) is a new run: its own directory, its own idle baseline
            string this_key = pi_mode_name(msg.mode) + "_seed" + std::to_string(msg.seed);
            if (run_key != this_key) {
                run_key = this_key;
                results_directory = output_directory + "/" + run_key;
                mkdir(results_directory.c_str(), 0755);
                if (save_genomes) {
                    mkdir((results_directory + "/genomes").c_str(), 0755);
                }
                string results_path = results_directory + "/pi_evaluations.csv";
                bool new_results = access(results_path.c_str(), F_OK) != 0;
                if (results.is_open()) {
                    results.close();
                }
                results.open(results_path, std::ios::app);
                if (new_results) {
                    results << "generation,mode,seed,island,genome_id,parameters,series,rows,mse,mae,naive_mse,build_ms,inference_ms,repeats,measured_ms,per_point_us,throughput_per_s,"
                            << "ina219_samples,bus_voltage_v_avg,current_ma_avg,power_mw_avg,idle_power_mw,energy_mj,energy_net_mj,energy_per_point_mj" << std::endl;
                }
                Log::info("=== run %s -> %s ===\n", run_key.c_str(), results_directory.c_str());

                // the pi has been idle waiting for this run's first message, so meter the
                // baseline now rather than reusing one taken at startup
                if (ina219_active) {
                    ina219_sampler.start(&ina219);
                    int64_t idle_start = INA219Sampler::now_us();
                    std::this_thread::sleep_for(std::chrono::milliseconds(idle_measure_ms));
                    int64_t idle_end = INA219Sampler::now_us();
                    ina219_sampler.stop();
                    INA219Stats idle = ina219_sampler.get_stats(idle_start, idle_end);
                    idle_power_mw = idle.power_mw_avg;
                    Log::info("idle baseline for %s over %d ms (%d readings, %.1f readings/s): %.3f V, %.1f mA, %.1f mW\n",
                        run_key.c_str(), idle_measure_ms, idle.sample_count, idle.sample_count * 1000.0 / idle_measure_ms,
                        idle.bus_voltage_v_avg, idle.current_ma_avg, idle_power_mw);
                    if (idle.sample_count < 10) {
                        Log::warning("INA219 delivered only %d readings in %d ms: energy numbers will be coarse\n", idle.sample_count, idle_measure_ms);
                    }
                }
            }

            if (msg.num_episodes_total != (int32_t) inputs.size()) {
                Log::error("generation %d: master has %d episodes but the pi sliced %d: the pi's data flags differ from the master's, skipping\n", g, msg.num_episodes_total, (int32_t) inputs.size());
                continue;
            }

            // this generation's test window
            Series3 test_inputs, test_outputs;
            bool ids_ok = true;
            for (int32_t id : msg.test_episode_ids) {
                if (id < 0 || id >= (int32_t) inputs.size()) {
                    Log::error("generation %d: test episode id %d out of range, skipping\n", g, id);
                    ids_ok = false;
                    break;
                }
                test_inputs.push_back(inputs[id]);
                test_outputs.push_back(outputs[id]);
            }
            if (!ids_ok || test_inputs.empty()) {
                continue;
            }

            vector<GenomeResult> gen_results;
            for (int32_t k = 0; k < (int32_t) msg.genome_bytes.size(); k++) {
                RNN_Genome* genome = new RNN_Genome(msg.genome_bytes[k].data(), (int32_t) msg.genome_bytes[k].size());
                GenomeResult r;
                r.island = msg.islands[k];
                r.genome_id = genome->get_generation_id();
                vector<double> parameters = genome->get_best_parameters();
                r.parameters = (int32_t) parameters.size();
                if (parameters.empty()) {
                    Log::warning("generation %d: genome %d has no best parameters, using its initial weights\n", g, r.genome_id);
                }

                if (save_genomes) {
                    genome->write_to_file(results_directory + "/genomes/generation_" + std::to_string(g) + "_island_" + std::to_string(r.island) + "_genome_" + std::to_string(r.genome_id) + ".bin");
                }

                // build: deserialized genome -> runnable network with its weights
                auto build_start = std::chrono::high_resolution_clock::now();
                RNN* rnn = genome->get_rnn();
                if (!parameters.empty()) {
                    rnn->set_weights(parameters);
                }
                r.build_ms = elapsed_ms(build_start);

                // inference: one forward pass per test series (per stock in pooled panel mode).
                // Repeated until the metered window is at least min_measure_ms long; the first
                // pass keeps the predictions, the time and energy are reported per pass.
                if (ina219_active) {
                    ina219_sampler.start(&ina219);
                }
                int64_t window_start = INA219Sampler::now_us();
                auto inference_start = std::chrono::high_resolution_clock::now();
                r.repeats = 0;
                do {
                    for (int32_t n = 0; n < (int32_t) test_inputs.size(); n++) {
                        vector<vector<double> > p = rnn->get_predictions(test_inputs[n], test_outputs[n], false, 0.0);
                        if (r.repeats == 0) {
                            r.predictions.push_back(p);
                        }
                    }
                    r.repeats++;
                } while (elapsed_ms(inference_start) < min_measure_ms);
                r.measured_ms = elapsed_ms(inference_start);
                int64_t window_end = INA219Sampler::now_us();
                r.inference_ms = r.measured_ms / r.repeats;
                if (ina219_active) {
                    ina219_sampler.stop();
                    r.power = ina219_sampler.get_stats(window_start, window_end);
                    r.power.energy_mj /= r.repeats;  // per inference
                    r.energy_net_mj = (r.power.power_mw_avg - idle_power_mw) * (r.inference_ms / 1000.0);
                } else {
                    r.power = INA219Stats();
                    r.energy_net_mj = 0.0;
                }
                delete rnn;
                delete genome;

                score(r.predictions, test_outputs, r.mse, r.mae, r.naive_mse, r.rows);
                gen_results.push_back(r);
            }

            // island_best: the ensemble is the mean prediction of the island bests
            if (msg.mode == PI_MODE_ISLAND_BEST && !gen_results.empty()) {
                GenomeResult e;
                e.island = -1;
                e.genome_id = -1;
                e.parameters = 0;
                e.build_ms = e.inference_ms = e.measured_ms = 0.0;
                e.repeats = 0;
                e.power = INA219Stats();
                e.energy_net_mj = 0.0;
                e.predictions = gen_results[0].predictions;
                for (auto& series : e.predictions) for (auto& outv : series) for (double& v : outv) v = 0.0;
                for (const GenomeResult& r : gen_results) {
                    e.parameters += r.parameters;
                    e.build_ms += r.build_ms;
                    e.inference_ms += r.inference_ms;   // the ensemble runs its members one after another
                    e.measured_ms += r.measured_ms;
                    e.repeats += r.repeats;
                    e.power.sample_count += r.power.sample_count;
                    e.power.energy_mj += r.power.energy_mj;
                    e.energy_net_mj += r.energy_net_mj;
                    e.power.bus_voltage_v_avg += r.power.bus_voltage_v_avg / gen_results.size();
                    e.power.current_ma_avg += r.power.current_ma_avg / gen_results.size();
                    e.power.power_mw_avg += r.power.power_mw_avg / gen_results.size();
                    for (size_t n = 0; n < r.predictions.size(); n++)
                        for (size_t i = 0; i < r.predictions[n].size(); i++)
                            for (size_t j = 0; j < r.predictions[n][i].size(); j++)
                                e.predictions[n][i][j] += r.predictions[n][i][j] / gen_results.size();
                }
                score(e.predictions, test_outputs, e.mse, e.mae, e.naive_mse, e.rows);
                gen_results.push_back(e);
            }

            for (const GenomeResult& r : gen_results) {
                bool ensemble = r.island < 0;
                double per_point_us = r.rows > 0 ? r.inference_ms * 1000.0 / r.rows : 0.0;
                double throughput = r.inference_ms > 0 ? r.rows / (r.inference_ms / 1000.0) : 0.0;
                if (ensemble) {
                    Log::info("generation %d ensemble of %d island bests: MSE %lf, MAE %lf (naive MSE %lf), inference %.2f ms total\n", g, (int32_t) gen_results.size() - 1, r.mse, r.mae, r.naive_mse, r.inference_ms);
                } else {
                    Log::info("generation %d island %d genome %d (%d params): MSE %lf, MAE %lf (naive MSE %lf), build %.2f ms, inference %.3f ms (%d repeats in %.1f ms), %.2f us/point, %.0f points/s\n", g, r.island, r.genome_id, r.parameters, r.mse, r.mae, r.naive_mse, r.build_ms, r.inference_ms, r.repeats, r.measured_ms, per_point_us, throughput);
                }
                if (ina219_active) {
                    if (r.power.sample_count > 0) {
                        Log::info("  INA219 (%d readings): %.3f V, %.1f mA, %.1f mW (idle %.1f mW); per inference %.4f mJ, net %.4f mJ, %.6f mJ/point\n", r.power.sample_count, r.power.bus_voltage_v_avg, r.power.current_ma_avg, r.power.power_mw_avg, idle_power_mw, r.power.energy_mj, r.energy_net_mj, r.rows > 0 ? r.power.energy_mj / r.rows : 0.0);
                    } else {
                        Log::warning("  INA219: no readings during the %.1f ms window\n", r.measured_ms);
                    }
                }
                results << g << "," << pi_mode_name(msg.mode) << "," << msg.seed << "," << r.island << "," << r.genome_id << "," << r.parameters << ","
                        << test_inputs.size() << "," << r.rows << "," << r.mse << "," << r.mae << "," << r.naive_mse << ","
                        << r.build_ms << "," << r.inference_ms << "," << r.repeats << "," << r.measured_ms << "," << per_point_us << "," << throughput << ","
                        << r.power.sample_count << "," << r.power.bus_voltage_v_avg << "," << r.power.current_ma_avg << ","
                        << r.power.power_mw_avg << "," << idle_power_mw << "," << r.power.energy_mj << "," << r.energy_net_mj << ","
                        << (r.rows > 0 ? r.power.energy_mj / r.rows : 0.0) << std::endl;
            }

            if (write_predictions && !gen_results.empty()) {
                string base = results_directory + "/generation_" + std::to_string(g);
                if (msg.mode == PI_MODE_ISLAND_BEST) {
                    Log::info("generation %d: writing %s_island_best.csv and %s_ensemble.csv\n", g, base.c_str(), base.c_str());
                    ofstream out(base + "_island_best.csv");
                    out << "island,elite_rank,stock,row,predicted\n";
                    for (const GenomeResult& r : gen_results) {
                        if (r.island < 0) continue;
                        for (int32_t n = 0; n < (int32_t) r.predictions.size(); n++)
                            for (int32_t j = 1; j < (int32_t) r.predictions[n][0].size(); j++)
                                out << r.island << ",0," << n << "," << (j - 1) << "," << r.predictions[n][0][j] << "\n";
                    }
                    write_global_format(base + "_ensemble.csv", "ensemble_predicted_", output_names, gen_results.back().predictions, test_outputs);
                } else {
                    Log::info("generation %d: writing %s_global_best.csv\n", g, base.c_str());
                    write_global_format(base + "_global_best.csv", "global_best_predicted_", output_names, gen_results[0].predictions, test_outputs);
                }
            }
        }

        Log::info("connection closed\n");
        close(fd);
    }

    if (ina219_active) {
        ina219.close_device();
    }
}
