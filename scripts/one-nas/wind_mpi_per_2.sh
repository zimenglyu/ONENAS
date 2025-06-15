#!/bin/sh
# This is an example of running ONE-NAS MPI version on wind turbine dataset
#
# The wind dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

#=============================================================================
# COMPREHENSIVE LIST OF ONE-NAS COMMAND LINE ARGUMENTS
#=============================================================================

# DATA INPUT/OUTPUT ARGUMENTS:
# --training_filenames <file1> <file2> ...    : CSV files for training data
# --test_filenames <file1> <file2> ...        : CSV files for test data
# --filenames <file1> <file2> ...             : CSV files (with training/test indexes)
# --training_indexes <int1> <int2> ...        : Indexes of training files (0-based)
# --test_indexes <int1> <int2> ...            : Indexes of test files (0-based)
# --input_parameter_names <name1> <name2> ... : Parameters to use as inputs
# --output_parameter_names <name1> <name2> ...: Parameters to use as outputs
# --shift_parameter_names <name1> <name2> ... : Parameters to shift to same timestep as output
# --time_offset <int>                          : Time offset for input/output data (default: 1)
# --output_directory <path>                    : Directory to save results and logs

# TIME SERIES PROCESSING:
# --time_series_length <int>                   : Length of time series sequences for slicing
# --train_sequence_length <int>               : Length for training sequences (if different)
# --validation_sequence_length <int>          : Length for validation sequences (if different)
# --normalize <type>                          : Normalization type: 'min_max', 'avg_std_dev', or 'none'

# ONLINE LEARNING ARGUMENTS:
# --num_training_sets <int>                   : Number of training sets per generation
# --num_validation_sets <int>                : Number of validation sets per generation
# --get_train_data_by <method>                : Training data selection: 'Uniform' or 'PER' (Prioritized Experience Replay)
# --start_score_tracking_generation <int>     : Generation to start tracking episode scores (default: 50)
# --total_generation <int>                    : Total number of generations to run

# EVOLUTION/SPECIATION ARGUMENTS:
# --speciation_method <method>                : Speciation strategy: 'island', 'neat', or 'onenas'
# --number_islands <int>                      : Number of islands for island-based methods
# --generated_population_size <int>           : Number of genomes generated per generation
# --elite_population_size <int>               : Number of elite genomes kept per island
# --island_size <int>                         : Size of each island population (for EXAMM)
# --max_genomes <int>                         : Maximum total genomes (for EXAMM)
# --num_mutations <int>                       : Maximum number of mutations per genome (default: 1)

# GENOME STRUCTURE ARGUMENTS:
# --possible_node_types <type1> <type2> ...   : Node types: simple, UGRNN, MGU, GRU, delta, LSTM
# --min_recurrent_depth <int>                 : Minimum recurrent depth (default: 1)
# --max_recurrent_depth <int>                 : Maximum recurrent depth (default: 10)

# TRAINING/BACKPROPAGATION ARGUMENTS:
# --bp_iterations <int>                       : Backpropagation iterations per genome
# --dropout_probability <float>               : Dropout probability for regularization
# --weight_update <method>                    : Weight update method: vanilla, momentum, nesterov, adagrad, rmsprop, adam, adam_bias
# --learning_rate <float>                     : Learning rate for backpropagation (default: 0.001)
# --high_threshold <float>                    : High gradient threshold (default: 1.0)
# --low_threshold <float>                     : Low gradient threshold (default: 0.05)

# WEIGHT UPDATE METHOD PARAMETERS:
# --mu <float>                                : Momentum parameter for momentum/nesterov methods (default: 0.9)
# --eps <float>                               : Epsilon parameter for adagrad/rmsprop/adam (default: 1e-8)
# --decay_rate <float>                        : Decay rate for rmsprop (default: 0.9)
# --beta1 <float>                             : Beta1 parameter for adam methods (default: 0.9)
# --beta2 <float>                             : Beta2 parameter for adam methods (default: 0.99)

# WEIGHT INITIALIZATION ARGUMENTS:
# --weight_initialize <method>                : Weight initialization: xavier, he, gaussian, uniform, lamarckian, gp
# --weight_inheritance <method>               : Weight inheritance: xavier, he, gaussian, uniform, lamarckian, gp
# --mutated_component_weight <method>         : Weight method for new components: xavier, he, gaussian, uniform, lamarckian, gp

# EXTINCTION/ISLAND MANAGEMENT:
# --extinction_event_generation_number <int> : Generation for extinction events
# --islands_to_exterminate <int>              : Number of islands to exterminate
# --island_ranking_method <method>            : Method for ranking islands
# --repopulation_method <method>              : Method for repopulating after extinction
# --repeat_extinction                         : Flag to repeat extinction events

# TRANSFER LEARNING:
# --transfer_learning_version <version>       : Transfer learning version
# --seed_stirs <int>                          : Number of seed stirs for transfer learning
# --start_filled                              : Flag to start with filled populations
# --tl_epigenetic_weights                     : Flag for transfer learning epigenetic weights

# NEAT-SPECIFIC ARGUMENTS:
# --species_threshold <float>                 : Species threshold for NEAT speciation
# --fitness_threshold <float>                 : Fitness threshold for NEAT (default: 100)
# --neat_c1 <float>                           : NEAT compatibility coefficient c1 (default: 1)
# --neat_c2 <float>                           : NEAT compatibility coefficient c2 (default: 1)
# --neat_c3 <float>                           : NEAT compatibility coefficient c3 (default: 1)

# GENOME LOADING/SAVING:
# --genome_bin <file>                         : Binary genome file to load
# --save_genome_option <option>               : Genome saving option: 'all_best_genomes', etc.
# --write_time_series <base_filename>         : Write processed time series to files

# LOGGING AND DEBUG:
# --std_message_level <level>                 : Standard output log level: FATAL, ERROR, WARNING, INFO, DEBUG, TRACE
# --file_message_level <level>                : File output log level: FATAL, ERROR, WARNING, INFO, DEBUG, TRACE
# --max_header_length <int>                   : Maximum header length for logging (default: 256)
# --max_message_length <int>                  : Maximum message length for logging (default: 1024)
# --generate_op_log                           : Flag to generate operation logs

# DNAS-SPECIFIC ARGUMENTS:
# --dnas_node_types <type1> <type2> ...       : Node types for DNAS evolution

# PERFORMANCE CONTROL ARGUMENTS:
# --compare_with_naive                        : Flag to enable naive vs genome prediction comparison
# --control_size_method <method>              : Network size control method: 'reduce_mutation_rate', 'reduce_add_mutation', 'none'

# ISLAND REPOPULATION ARGUMENTS:
# --repopulation_frequency <int>              : Frequency for island repopulation events

# --temperature <float>                        : Tempered sampling temperature τ for PER (default: 1.0)

#=============================================================================

cd build

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

for i in {0..4}
do

exp_name="../results_2/wind_per/$i"
mkdir -p $exp_name
echo "Running ONE-NAS with NEW EPISODE MANAGEMENT system on wind turbine dataset"
echo "Results will be saved to: "$exp_name


mpirun -np 16 ./mpi/onenas_mpi \
--training_filenames ../datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-9].csv  ../datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 20 \
--bp_iterations 10 \
--output_directory $exp_name \
--num_mutations 1 \
--time_series_length 25 \
--num_validation_sets 50 \
--num_training_sets 300  \
--get_train_data_by PER \
--speciation_method onenas \
--repopulation_frequency 50 \
--generated_population_size 10 \
--elite_population_size 10 \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--start_score_tracking_generation 500 \
--normalize min_max \
--compare_with_naive \
--control_size_method reduce_mutation_rate \
--std_message_level INFO \
--file_message_level INFO \
--temperature 1.0 

done