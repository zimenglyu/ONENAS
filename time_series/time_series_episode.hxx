#ifndef TIME_SERIES_EPISODE_HXX
#define TIME_SERIES_EPISODE_HXX

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

class TimeSeriesEpisode {
   private:
    int32_t episode_id; // episode id is the index of the episode in the original time series - this NEVER changes
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;
    
    // Episode-specific training metrics
    int32_t training_score;
    // vector<int32_t> training_generations;  // generations this episode was used for training
    // map<int32_t, int32_t> genome_performance;  // genome_id -> performance score
    
    // //  TODO: Memory management and access tracking
    bool is_loaded;
    // int32_t access_count;
    // double importance_score;
    
    // // TODO: For future disk-based storage (optional)
    // string data_file_path;
    
    // void update_access_time();

   public:
    // Constructors
    TimeSeriesEpisode(int32_t id);
    TimeSeriesEpisode(int32_t id, const vector<vector<double>>& inputs, const vector<vector<double>>& outputs);
    
    // Destructor
    ~TimeSeriesEpisode();
    
    // Data access methods
    const vector<vector<double>>& get_inputs();
    const vector<vector<double>>& get_outputs();
    void set_data(const vector<vector<double>>& inputs, const vector<vector<double>>& outputs);
    
    // Scoring and training history methods
    void update_training_score(int32_t increment = 1);
    int32_t get_training_score() const;
    void add_training_generation(int32_t generation_id);
    void update_genome_performance(int32_t genome_id, int32_t score);
    const vector<int32_t>& get_training_generations() const;
    
    // Memory management
    bool is_data_loaded() const;
    void ensure_loaded();  // Make sure data is in memory
    void unload_data();    // Remove data from memory (for future memory management)
    
    // Access tracking
    int32_t get_access_count() const;
    
    // Importance calculation for retention policies
    double calculate_importance(int32_t current_generation = -1) const;
    bool should_keep(int32_t current_generation, int32_t max_age = 100) const;
    
    // Episode identification
    int32_t get_episode_id() const;
    
    // For debugging
    void print_stats() const;
    
    // Future serialization support (stubs for now)
    void save_to_file(const string& filepath);
    void load_from_file(const string& filepath);
};

#endif 