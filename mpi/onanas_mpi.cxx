#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "onenas/onenas.hxx"
#include "mpi.h"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "time_series/online_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

#define WORK_REQUEST_TAG  1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG        3
#define TERMINATE_TAG     4

mutex onenas_mutex;

vector<string> arguments;

ONENAS* onenas;
WeightUpdate* weight_update_method;

bool finished = false;

vector<vector<vector<double> > > time_series_inputs;
vector<vector<vector<double> > > time_series_outputs;
vector<int32_t> time_series_index;
int32_t generated_population_size;
int32_t number_islands;
int32_t total_generation;

/**
 * Checks if enough genomes have been generated for the current generation
 * 
 * @param current_generated_genomes The number of genomes generated so far
 * @return true if enough genomes have been generated, false otherwise
 */
bool has_generated_enough_genomes(int32_t current_generated_genomes) {
    return current_generated_genomes >= generated_population_size * number_islands;
}

void send_work_request(int32_t target) {
    int32_t work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int32_t source) {
    MPI_Status status;
    int32_t work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int32_t source) {
    MPI_Status status;
    int32_t length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int32_t length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void send_genome_to(int32_t target, RNN_Genome* genome) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int32_t target) {
    int32_t terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int32_t source) {
    MPI_Status status;
    int32_t terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}

void master(int32_t max_rank) {
    // the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX int32_t: %d\n", numeric_limits<int32_t>::max());

    int32_t terminates_sent = 0;
    int32_t generated_genome = 0;
    int32_t evaluated_genome = 0;
    while (true) {
        // wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);
            if (!has_generated_enough_genomes(generated_genome)) {
                onenas_mutex.lock();
                RNN_Genome *genome = onenas->generate_genome();
                onenas_mutex.unlock();

                if (genome != NULL) {
                    Log::info("genome is not null, Sending genome to worker: %d\n", source);
                    Log::debug("sending genome to: %d\n", source);
                    send_genome_to(source, genome);

                    //delete this genome as it will not be used again
                    delete genome;
                    generated_genome ++;
                } else {
                    Log::fatal("Returned NULL genome from generate genome function, this should never happen!\n");
                    exit(1);
                }
            } else {
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::info("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    Log::debug("Ending genome, generated genome is %d, evaluated genome is %d\n", generated_genome, evaluated_genome);
                    return;
                }
            }

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome* genome = receive_genome_from(source);

            onenas_mutex.lock();
            onenas->insert_genome(genome);
            onenas_mutex.unlock();

            // delete the genome as it won't be used again, a copy was inserted
            delete genome;
            evaluated_genome++;
            // this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int32_t rank, OnlineSeries* online_series) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int32_t tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::info("worker %d received genome!\n", rank);
            RNN_Genome* genome = receive_genome_from(0);

            vector< vector< vector<double> > > current_training_inputs;
            vector< vector< vector<double> > > current_training_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            vector<int32_t> train_index = online_series->get_training_index();
            vector<int32_t> validation_index = online_series->get_validation_index();

            for ( int32_t  i = 0; i < (int32_t)train_index.size(); i++) {
                current_training_inputs.push_back(time_series_inputs[train_index[i]]);
                current_training_outputs.push_back(time_series_outputs[train_index[i]]);
                Log::debug("Worker: training index: %d\n", train_index[i]);
            }
            for (int32_t  i = 0; i < (int32_t)validation_index.size(); i++) {
                current_validation_inputs.push_back(time_series_inputs[validation_index[i]]);
                current_validation_outputs.push_back(time_series_outputs[validation_index[i]]);
                Log::debug("Worker: validation index: %d\n", validation_index[i]);
            }

            //have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(current_training_inputs, current_training_outputs, current_validation_inputs, current_validation_outputs, weight_update_method);
            // genome->set_genome_type(GENERATED);
            genome->evaluate_online(current_validation_inputs, current_validation_outputs);
            Log::release_id(log_id);

            // go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

int main(int argc, char** argv) {
    std::cout << "Starting ONENAS MPI Program" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "MPI initialized" << std::endl;
    int32_t rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
    std::cout << "MPI got rank " << rank << " and max rank " << max_rank << std::endl;
    arguments = vector<string>(argv, argv + argc);
    std::cout << "Received arguments!" << std::endl;

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);
    std::cout << "Initialized log!" << std::endl;

    // Load required parameters
    get_argument(arguments, "--number_islands", true, number_islands);
    get_argument(arguments, "--generated_population_size", true, generated_population_size);

    Log::info("ONENAS will generate %d genomes per generation\n", generated_population_size * number_islands);

    TimeSeriesSets* time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    slice_online_time_series(
        arguments, time_series_sets, time_series_inputs, time_series_outputs
    );
    Log::major_divider(Log::INFO, "Sliced time series!");
    Log::info("Time series inputs shape: %d, %d, %d \n", time_series_inputs.size(), time_series_inputs[0].size(), time_series_inputs[0][0].size());
    Log::info("Time series outputs shape: %d, %d, %d \n", time_series_outputs.size(), time_series_outputs[0].size(), time_series_outputs[0][0].size());
    int32_t num_sets = time_series_inputs.size();
    Log::info("Time series number of sets after slicing: %d\n", num_sets);
    total_generation = num_sets; // default to the number of sets
    get_argument(arguments, "--total_generation", false, total_generation); 
    if (total_generation > num_sets) {
        Log::error("Total generation is greater than the number of sets, setting total generation to %d\n", num_sets);
        total_generation = num_sets;
    }
    
    OnlineSeries* online_series = new OnlineSeries(num_sets, arguments);

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);
    Log::major_divider(Log::INFO, "Created weight update method!");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);
    if (weight_rules == NULL) {
        Log::fatal("ERROR in onenas mpi: Failed to create weight rules, this should not happen\n");
        exit(1);
    }
    Log::major_divider(Log::INFO, "Created weight rules!");

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);
    Log::major_divider(Log::INFO, "Created seed genome!");

    Log::clear_rank_restriction();

    if (rank == 0) {
        onenas = generate_onenas_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);
        Log::major_divider(Log::INFO, "Created ONENAS!");
    }

    for (int32_t  current_generation = 0; current_generation < num_sets; current_generation ++) {
        online_series->set_current_index(current_generation);
        // Log::info("current generation: %d\n", current_generation);


        if (rank ==0) {
            Log::minor_divider(Log::INFO);
            Log::info("Current generation: %d \n", current_generation);
            master(max_rank);           
        } else {
            worker(rank, online_series);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            vector <int32_t> validation_index = online_series->get_validation_index();
            int32_t test_index = online_series->get_test_index();

            vector< vector< vector<double> > > current_test_inputs;
            vector< vector< vector<double> > > current_test_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            current_test_inputs.push_back(time_series_inputs[test_index]);
            current_test_outputs.push_back(time_series_outputs[test_index]);

            for (int32_t  i = 0; i < (int32_t)validation_index.size(); i++) {
                current_validation_inputs.push_back(time_series_inputs[validation_index[i]]);
                current_validation_outputs.push_back(time_series_outputs[validation_index[i]]);
                Log::debug("validation index: %d\n", validation_index[i]);
            }
            Log::info("Current testing index: %d\n", test_index);

            onenas->finalize_generation(current_generation, current_validation_inputs, current_validation_outputs, current_test_inputs, current_test_outputs);
            onenas->update_log();

        }


        if (rank == 0) Log::error("generation %d finished\n", current_generation);
        
    }
    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    return 0;
}
