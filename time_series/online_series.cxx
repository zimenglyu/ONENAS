#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <iostream>
using std::ios;

#include <sys/stat.h>

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
    cleanup_frequency = 0; // default: no cleanup
    get_online_arguments(arguments);
    
    // Initialize episodes vector
    episodes.reserve(total_num_sets);
}

OnlineSeries::~OnlineSeries() {
    // Episodes will be automatically cleaned up by unique_ptr destructors
    episodes.clear();
}

void OnlineSeries::get_online_arguments(const vector<string> &arguments) {
    get_argument(arguments, "--num_validataion_sets", true, num_validataion_sets);
    get_argument(arguments, "--num_training_sets", true, num_training_sets);
    get_argument(arguments, "--get_train_data_by", true, get_training_data_method);
    start_score_tracking_generation = num_training_sets; // default value
    get_argument(arguments, "--start_score_tracking_generation", false, start_score_tracking_generation);
    get_argument(arguments, "--cleanup_frequency", false, cleanup_frequency);
}

void OnlineSeries::set_current_index(int32_t _current_gen) {
    //current index is the begining of validation index
    current_index = _current_gen + num_training_sets;
    Log::debug("current generation is %d, current index is %d\n", _current_gen, current_index);
}

void OnlineSeries::shuffle_data() {
    // current index is the end of available training index
    // avalibale_training_index contains episode IDs (original time series indices)
    avalibale_training_index.clear();

    for (int32_t i = 0; i < current_index; i++) {
        avalibale_training_index.push_back(i);  // i is the episode ID (original time series index)
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
        // Get episode by ID (idx is the original episode ID)
        int32_t score = get_episode_training_score(idx);
        available_scores.push_back(score);
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
        int32_t actual_episode_id = avalibale_training_index[sampled_idx];  // Get the actual episode ID
        
        // To avoid repeats
        if (seen.find(actual_episode_id) == seen.end()) {
            training_index.push_back(actual_episode_id);
            seen.insert(actual_episode_id);
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
    // train_index contains episode IDs (original time series indices)
    vector<int32_t> train_index_copy = train_index;
    training_history[generation_id] = train_index_copy;
    Log::info("OnlineSeries: Added training history for generation_id: %d with %d episode IDs\n", 
             generation_id, (int32_t)train_index.size());
    Log::debug("OnlineSeries: Current training_history map size: %d\n", (int32_t)training_history.size());
}

vector<int32_t> OnlineSeries::get_training_history(int32_t generation_id) {
    Log::info("Getting training history for generation id: %d\n", generation_id);
    return training_history[generation_id];
}

void OnlineSeries::update_scores(vector<int32_t>& generation_ids, int32_t current_generation) {
    Log::info("OnlineSeries: Starting score updates for generation %d\n", current_generation);
    
    // Check if we should start updating scores based on current generation
    if (current_generation < start_score_tracking_generation) {
        Log::info("Generation %d is before score tracking threshold (%d), skipping score updates\n", 
                 current_generation, start_score_tracking_generation);
        return;
    }

    Log::info("Generation %d >= threshold (%d), updating scores for %d good genomes\n", 
             current_generation, start_score_tracking_generation, (int32_t)generation_ids.size());

    // generation_ids contains the generation IDs of good genomes (elite genomes from islands)
    // For each good genome, we look up which episodes it used for training and increment their scores by +1
    update_episode_scores(generation_ids, current_generation);
    Log::info("Completed episode score updates\n");
}

void OnlineSeries::write_scores_to_csv(int32_t generation, const string& stats_directory) {
    // Create full path for the CSV file (stats directory already includes the stats path)
    string csv_file_path = stats_directory + "/training_scores.csv";
    
    // Check if file exists to determine if we need to write header
    bool file_exists = false;
    {
        std::ifstream test_file(csv_file_path);
        file_exists = test_file.good();
    }
    
    // Open CSV file in append mode
    ofstream csv_file(csv_file_path, ios::app);
    
    if (!csv_file.is_open()) {
        Log::error("Failed to open %s for writing\n", csv_file_path.c_str());
        return;
    }
    
    // Write header if file is new - use total_num_sets for all episodes
    if (!file_exists) {
        csv_file << "generation";
        for (int32_t episode_id = 0; episode_id < total_num_sets; episode_id++) {
            csv_file << ",episode_" << (episode_id + 1);  // 1-indexed episode names
        }
        csv_file << "\n";
    }
    
    // Write generation number as first column
    csv_file << generation;
    
    // Write scores for all episodes up to total_num_sets
    for (int32_t episode_id = 0; episode_id < total_num_sets; episode_id++) {
        int32_t score = get_episode_training_score(episode_id);
        csv_file << "," << score;
    }
    
    // End the row
    csv_file << "\n";
    csv_file.close();
    
    Log::info("Written scores for generation %d to %s\n", generation, csv_file_path.c_str());
}

// New episode management methods

void OnlineSeries::add_episode(unique_ptr<TimeSeriesEpisode> episode) {
    episodes.push_back(std::move(episode));
}

void OnlineSeries::initialize_episodes(const vector<vector<vector<double>>>& inputs, const vector<vector<vector<double>>>& outputs) {
    episodes.clear();
    int32_t num_episodes = min(inputs.size(), outputs.size());
    
    for (int32_t i = 0; i < num_episodes; i++) {
        // c++ 11 smart pointer to prevent memory leak
        auto episode = std::make_unique<TimeSeriesEpisode>(i, inputs[i], outputs[i]);
        episodes.push_back(std::move(episode));
    }
    
    Log::info("Initialized %d episodes\n", num_episodes);
}

TimeSeriesEpisode* OnlineSeries::get_episode(int32_t episode_id) {
    // Search for episode by ID, not by vector index
    for (int32_t i = 0; i < (int32_t)episodes.size(); i++) {
        if (episodes[i] && episodes[i]->get_episode_id() == episode_id) {
            return episodes[i].get();
        }
    }
    return nullptr;
}

void OnlineSeries::print_episode_stats() {
    Log::info("Episode Statistics:\n");
    Log::info("Total episodes: %d\n", (int32_t)episodes.size());
    for (int32_t i = 0; i < min(5, (int32_t)episodes.size()); i++) {
        episodes[i]->print_stats();
    }
}

void OnlineSeries::update_episode_scores(vector<int32_t>& generation_ids, int32_t current_generation) {
    Log::info("OnlineSeries: Updating episode scores for %d good genomes (generation IDs)\n", 
             (int32_t)generation_ids.size());
    
    for (int32_t i = 0; i < (int32_t)generation_ids.size(); i++) {
        int32_t generation_id = generation_ids[i];  // This is the good genome's generation ID
        vector<int32_t> train_episode_ids = get_training_history(generation_id);
        
        if (train_episode_ids.size() == 0) {
            Log::error("No training history found for generation id: %d\n", generation_id);
            continue;
        }
        
        Log::debug("Incrementing scores for %d episodes used by good genome %d\n", 
                  (int32_t)train_episode_ids.size(), generation_id);
        
        // For each episode ID that this good genome used for training, increment its score by +1
        for (int32_t j = 0; j < (int32_t)train_episode_ids.size(); j++) {
            int32_t episode_id = train_episode_ids[j];  // This is the original episode ID (time series index)
            
            // Find episode by ID and increment its score
            for (int32_t k = 0; k < (int32_t)episodes.size(); k++) {
                if (episodes[k] && episodes[k]->get_episode_id() == episode_id) {
                    episodes[k]->update_training_score(1);  // Add score by +1
                    episodes[k]->add_training_generation(generation_id);
                    Log::debug("Incremented score for episode ID %d (used by good genome %d)\n", 
                              episode_id, generation_id);
                    break;  // Found the episode, no need to continue searching
                }
            }
        }
    }
}

int32_t OnlineSeries::get_episode_training_score(int32_t episode_id) {
    // Search for episode by ID, not by vector index
    for (int32_t i = 0; i < (int32_t)episodes.size(); i++) {
        if (episodes[i] && episodes[i]->get_episode_id() == episode_id) {
            return episodes[i]->get_training_score();
        }
    }
    return 1; // Default score
}

bool OnlineSeries::should_cleanup_episodes(int32_t current_generation) {
    // No cleanup if frequency is 0 or negative
    if (cleanup_frequency <= 0) {
        return false;
    }
    
    // Cleanup if current generation is a multiple of cleanup_frequency
    return (current_generation % cleanup_frequency == 0);
}

void OnlineSeries::perform_periodic_cleanup(int32_t current_generation) {
    if (should_cleanup_episodes(current_generation)) {
        Log::info("Periodic cleanup requested at generation %d, but cleanup functionality has been removed\n", current_generation);
        // print_episode_stats();
    }
}