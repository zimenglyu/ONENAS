#ifndef ONLINE_SERIES_HXX
#define ONLINE_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

#include <unordered_set>
using std::unordered_set;

#include <random>
using std::normal_distribution;
using std::default_random_engine;

#include "time_series_episode.hxx"

class OnlineSeries {
    private:
        // Episode management - new approach
        vector<TimeSeriesEpisode*> episodes;
        
        // // Legacy members for compatibility (will be gradually phased out)
        // vector< double > mean;
        // vector< double > std;
        // normal_distribution<double> gaussian;
        // default_random_engine noise_generator;
        
        // Core configuration
        int32_t total_num_sets;
        int32_t sequence_length;
        int32_t current_index; // current index = current_generation + num_training_sets
        vector< int32_t > avalibale_training_index;
        int32_t num_training_sets;
        int32_t num_validation_sets;
        int32_t num_test_sets;
        int32_t start_score_tracking_generation; // generation number to start updating scores, default 20
        string get_training_data_method;
        map<int, vector<int32_t>> training_history; // key: generation id, value: training index
        
        // Tempered Sampling parameters for PER
        double temperature; // τ (tau) for tempered sampling: P(x_i) = w_i^(1/τ) / Σw_j^(1/τ)
        
    public:
        OnlineSeries(int32_t _num_sets, const vector<string> &arguments);
        ~OnlineSeries();

        // Episode management methods - new approach
        void add_episode(TimeSeriesEpisode* episode);
        void initialize_episodes(const vector<vector<vector<double>>>& inputs, const vector<vector<vector<double>>>& outputs);
        TimeSeriesEpisode* get_episode(int32_t episode_id);
        void print_episode_stats();
        
        // Smart memory management method based on good genome IDs
        void cleanup_old_training_history(const vector<int32_t>& good_genome_ids);
        
        // Legacy methods (maintained for compatibility)
        void get_mean_std();
        void shuffle_data();
        void uniform_random_sample_index(vector<int32_t>& training_index);
        void prioritized_experience_replay(vector<int32_t>& training_index);
        void validate_generation_number(int32_t num_generation);
        void set_current_index(int32_t _current_gen);
        void get_online_arguments(const vector<string> &arguments);
        void add_gaussian_noise(vector< vector<double> > &noisy_values, double noise_percent);
        void slice_series_to_sets();
        
        // Core interface methods (updated to work with episodes)
        vector<int32_t> get_training_index(vector<int32_t>& training_index);
        vector< int32_t > get_validation_index(vector<int32_t>& validation_index);
        int32_t get_test_index();
        void add_score(int32_t index, int32_t score);
        void add_training_history(int32_t generation_id, vector<int32_t>& train_index);
        vector<int32_t> get_training_history(int32_t generation_id);
        void update_scores(vector<int32_t>& generation_ids, int32_t current_generation);
        void write_scores_to_csv(int32_t generation, const string& stats_directory);
        
        // New episode-specific methods
        void update_episode_scores(vector<int32_t>& generation_ids, int32_t current_generation);
        int32_t get_episode_training_score(int32_t episode_id);
        
        // Getter for training data method
        string get_training_method() const { return get_training_data_method; }
        
        // Memory monitoring helper
        int32_t get_training_history_size() const { return (int32_t)training_history.size(); }

        int32_t get_max_generation();
};



#endif