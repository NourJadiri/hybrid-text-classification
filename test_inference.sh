#!/bin/bash
# test_inference.sh - Test script for the batch inference functionality

set -e

echo "========================================"
echo "TESTING BATCH INFERENCE SCRIPT"
echo "========================================"

# Check if required directories exist
if [ ! -d "devset/EN/subtask-2-documents" ]; then
    echo "Error: Test directory devset/EN/subtask-2-documents not found!"
    exit 1
fi

# Create a small test folder with a few files
TEST_DIR="test_inference_sample"
mkdir -p $TEST_DIR

echo "Creating test samples..."
cp devset/EN/subtask-2-documents/EN_CC_200030.txt $TEST_DIR/
cp devset/EN/subtask-2-documents/EN_CC_200033.txt $TEST_DIR/
cp devset/EN/subtask-2-documents/EN_UA_DEV_100003.txt $TEST_DIR/ 2>/dev/null || echo "File EN_UA_DEV_100003.txt not found, skipping"

# Find any two files if the specific ones don't exist
if [ $(ls $TEST_DIR/*.txt 2>/dev/null | wc -l) -lt 2 ]; then
    echo "Getting first 3 files from devset..."
    rm -f $TEST_DIR/*.txt 2>/dev/null || true
    ls devset/EN/subtask-2-documents/*.txt | head -3 | xargs -I {} cp {} $TEST_DIR/
fi

echo "Test files prepared:"
ls -la $TEST_DIR/

echo ""
echo "========================================"
echo "TESTING SINGLE FILE INFERENCE"
echo "========================================"

FIRST_FILE=$(ls $TEST_DIR/*.txt | head -1)
echo "Testing single file: $FIRST_FILE"

# You would run this with an actual model path
echo "Command that would be run:"
echo "python src/scripts/simple_inference.py --file '$FIRST_FILE' --model_path 'YOUR_MODEL_PATH' --prompt_type constrained"

echo ""
echo "========================================"
echo "TESTING BATCH INFERENCE"
echo "========================================"

echo "Command that would be run for batch processing:"
echo "python src/scripts/batch_inference.py --folder '$TEST_DIR' --model_path 'YOUR_MODEL_PATH' --prompt_type constrained --output 'test_predictions.txt'"

echo ""
echo "Or using the convenience script:"
echo "./run_batch_inference.sh 'YOUR_MODEL_PATH' '$TEST_DIR' constrained test_predictions.txt"

echo ""
echo "========================================"
echo "EXAMPLE MODEL PATHS TO USE:"
echo "========================================"
echo "If you have a finetuned Qwen model:"
echo "  models/qwen-1.7b-5-epochs/final_model"
echo "  models/qwen-finetuned/final_model"
echo ""
echo "Replace 'YOUR_MODEL_PATH' with the actual path to your model"

# Clean up
echo ""
echo "Cleaning up test directory..."
rm -rf $TEST_DIR

echo "Test preparation completed!"
echo ""
echo "To run actual inference, use:"
echo "1. For single file: python src/scripts/simple_inference.py --file <file_path> --model_path <model_path>"
echo "2. For batch: python src/scripts/batch_inference.py --folder <folder_path> --model_path <model_path>"
echo "3. Using convenience script: ./run_batch_inference.sh <model_path> <folder_path>"
