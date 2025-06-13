#!/bin/bash

# Script to evaluate RNN with specified genome and test files
# Usage: ./evaluate_rnn.sh

# Set script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
GENOME_FILE="/Users/zimenglyu/Documents/code/git/ONE-NAS/online_test_output/wind_mpi/rnn_genome_46860.bin"
TEST_DATA_DIR="/Users/zimenglyu/Documents/code/git/ONE-NAS/online_test_output/wind_data"
OUTPUT_DIR="./prediction_results"
EXECUTABLE="$PROJECT_ROOT/build/rnn_examples/evaluate_rnn"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please build the project first using cmake and make"
    exit 1
fi

# Check if genome file exists
if [ ! -f "$GENOME_FILE" ]; then
    echo "Error: Genome file not found at $GENOME_FILE"
    exit 1
fi

echo "Running RNN evaluation..."
echo "Genome file: $GENOME_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Processing files from generation_60.csv to generation_520.csv"
echo ""

# Process each file individually
FILES_PROCESSED=0
for i in $(seq 60 520); do
    FILE_PATH="$TEST_DATA_DIR/generation_${i}.csv"
    if [ -f "$FILE_PATH" ]; then
        echo "Processing: generation_${i}.csv"
        
        # Run the evaluation for this file
        "$EXECUTABLE" \
            --output_directory "$OUTPUT_DIR" \
            --genome_file "$GENOME_FILE" \
            --testing_filenames "$FILE_PATH" \
            --time_offset 1 \
            --std_message_level INFO \
            --file_message_level INFO
            
        FILES_PROCESSED=$((FILES_PROCESSED + 1))
    else
        echo "Warning: Test file not found: $FILE_PATH"
    fi
done

# Check if we processed any files
if [ $FILES_PROCESSED -eq 0 ]; then
    echo "Error: No test files found in the range generation_60.csv to generation_520.csv"
    exit 1
fi

echo ""
echo "Processed $FILES_PROCESSED files."

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_DIR" 