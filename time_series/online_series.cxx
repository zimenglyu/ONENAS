#include <string>
using std::string;

#include <vector>
using std::vector;

using std::min;

#include <algorithm> 
using std::shuffle;

#include <random>
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;

#include <unordered_set>
using std::unordered_set;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "online_series.hxx"

OnlineSeries::OnlineSeries(const int32_t _total_num_sets,const vector<string> &arguments) {
    total_num_sets = _total_num_sets;
    current_index = 0;
    start_score_tracking_generation = 50; // default value
    get_online_arguments(arguments);
    training_scores.resize(total_num_sets, 1);
}

void OnlineSeries::get_online_arguments(const vector<string> &arguments) {
    get_argument(arguments, "--num_validataion_sets", true, num_validataion_sets);
    get_argument(arguments, "--num_training_sets", true, num_training_sets);
    get_argument(arguments, "--get_train_data_by", true, get_training_data_method);
    start_score_tracking_generation = num_training_sets; // default value
    get_argument(arguments, "--start_score_tracking_generation", false, start_score_tracking_generation);
    // get_argument(arguments, "--time_series_length", true, sequence_length);
    // get_argument(arguments, "--generation_genomes", true, generation_genomes);
    // get_argument(arguments, "--elite_population_size", true, elite_population_size);
}

void OnlineSeries::set_current_index(int32_t _current_gen) {
    //current index is the begining of validation index
    current_index = _current_gen + num_training_sets;
    Log::debug("current generation is %d, current index is %d\n", _current_gen, current_index);
}

void OnlineSeries::shuffle_data() {
    // current index is the end of available training index
    avalibale_training_index.clear();

    for (int32_t i = 0; i < current_index; i++) {
        avalibale_training_index.push_back(i);
    }

    auto rng = std::default_random_engine {};
    shuffle(avalibale_training_index.begin(), avalibale_training_index.end(), rng);
}

void OnlineSeries::uniform_random_sample_index(vector<int32_t>& training_index) {
    shuffle_data();
    training_index.clear();
    for (int32_t i = 0; i < num_training_sets; i++) {
        training_index.push_back(avalibale_training_index[i]);
    }
}

void OnlineSeries::prioritized_experience_replay(vector<int32_t>& training_index) {
    shuffle_data();
    training_index.clear();

    // Create scores vector for available indices only
    vector<int32_t> available_scores;
    for (int32_t idx : avalibale_training_index) {
        available_scores.push_back(training_scores[idx]);
    }

    // Random number generator
    std::random_device rd;
    // High-quality, fast random number generation. Standard modern RNG engine in C++11+.
    std::mt19937 gen(rd());

    // Use scores as weights - creates a distribution where each index is chosen with probability proportional to its weight.
    std::discrete_distribution<> dist(available_scores.begin(), available_scores.end());

    // To avoid duplicates
    std::unordered_set<int32_t> seen;

    while ((int32_t)training_index.size() < num_training_sets) {
        int32_t sampled_idx = dist(gen);  // This is an index into available_scores/avalibale_training_index
        int32_t actual_training_idx = avalibale_training_index[sampled_idx];  // Get the actual training index
        
        // To avoid repeats
        if (seen.find(actual_training_idx) == seen.end()) {
            training_index.push_back(actual_training_idx);
            seen.insert(actual_training_idx);
        }
        // Log::info("current training index size: %d\n", (int32_t)training_index.size());
    }
}

vector<int32_t> OnlineSeries::get_training_index(vector<int32_t>& training_index) {

    if (get_training_data_method.compare("Uniform") == 0) {
        Log::info("getting historical data with uniform random sampling\n");
        // V1 means all the generated genome has different random historical data
        // int32_t s = min(num_training_sets, current_index);
        uniform_random_sample_index(training_index);
    } else if (get_training_data_method.compare("PER") == 0) {
        Log::info("getting historical data with Priotized experience replay\n");
        // int32_t num_random_sets = min(num_training_sets, current_index);
        prioritized_experience_replay(training_index);
    } else {
        Log::error("Invalid training data method: %s\n", get_training_data_method.c_str());
        exit(1);
    }

    return training_index;
    
}

vector< int32_t > OnlineSeries::get_validation_index(vector<int32_t>& validation_index) {
    validation_index.clear();
    for (int32_t i = 0; i < num_validataion_sets; i++) {
        validation_index.push_back(current_index + i);
    }
    return validation_index;
}

int32_t OnlineSeries::get_test_index() {
    return current_index + num_validataion_sets;
}

void OnlineSeries::add_training_history(int32_t generation_id, vector<int32_t>& train_index) {
    // Make an explicit copy of train_index since it will be deleted after this function call
    vector<int32_t> train_index_copy = train_index;
    training_history[generation_id] = train_index_copy;
    Log::info("OnlineSeries instance address: %p - Added training history for generation_id: %d with %d training indices\n", this, generation_id, (int32_t)train_index.size());
    Log::info("OnlineSeries instance address: %p - Current training_history map size: %d\n", this, (int32_t)training_history.size());
}

vector<int32_t> OnlineSeries::get_training_history(int32_t generation_id) {
    Log::info("Getting training history for generation id: %d\n", generation_id);
    return training_history[generation_id];
}

void OnlineSeries::update_scores(vector<int32_t>& generation_ids, int32_t current_generation) {
    Log::info("Updating scores for generation ids: \n");
    
    // Check if we should start updating scores based on current generation
    if (current_generation < start_score_tracking_generation) {
        Log::info("Generation %d is before score tracking threshold (%d), skipping score updates\n", 
                 current_generation, start_score_tracking_generation);
        return;
    }

    Log::info("Generation %d >= threshold (%d), updating scores for good genomes\n", 
             current_generation, start_score_tracking_generation);

    for (int32_t i = 0; i < (int32_t)generation_ids.size(); i++) {
        int32_t generation_id = generation_ids[i];
        vector<int32_t> train_index = get_training_history(generation_id);
        if (train_index.size() == 0) {
            Log::error("No training history found for generation id: %d\n", generation_id);
            exit(1);
        }
        for (int32_t i = 0; i < (int32_t)train_index.size(); i++) {
            training_scores[train_index[i]]++;
        }
    }
    Log::info("Updated training history \n");
}

void OnlineSeries::print_scores() {
    Log::info("Current training scores: \n");
    for (int32_t i = 0; i < current_index; i++) {
        Log::info("%d \n", training_scores[i]);
    }
    Log::info("\n");
}