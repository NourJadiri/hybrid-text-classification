#!/bin/bash
# run_batch_inference.sh

# This script runs batch inference on a folder of text files
# Usage: ./run_batch_inference.sh <model_path> <input_folder> [prompt_type] [output_file]

set -e

# Default values
PROMPT_TYPE=${3:-"constrained"}
OUTPUT_FILE=${4:-""}

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <input_folder> [prompt_type] [output_file]"
    echo ""
    echo "Arguments:"
    echo "  model_path    : Path to the finetuned model"
    echo "  input_folder  : Path to folder containing .txt files"
    echo "  prompt_type   : Type of prompt (constrained, simple, comprehensive) [default: constrained]"
    echo "  output_file   : Output file path [default: auto-generated]"
    echo ""
    echo "Examples:"
    echo "  $0 models/qwen-1.7b-5-epochs devset/EN constrained EN_predictions.txt"
    echo "  $0 models/phase1_xlmr_augmented_best_model.bin testset/RU simple"
    exit 1
fi

MODEL_PATH=$1
INPUT_FOLDER=$2

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# Check if model path exists (skip check for Hugging Face model names)
if [[ "$MODEL_PATH" != *"/"* ]] || [[ "$MODEL_PATH" == *"models/"* ]]; then
    # Only check existence for local paths
    if [ ! -e "$MODEL_PATH" ]; then
        echo "Error: Model path '$MODEL_PATH' does not exist."
        exit 1
    fi
fi

echo "========================================"
echo "BATCH INFERENCE SCRIPT"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Input Folder: $INPUT_FOLDER"
echo "Prompt Type: $PROMPT_TYPE"
echo "Output File: ${OUTPUT_FILE:-"Auto-generated"}"
echo "========================================"

# Build the command
CMD=" source .venv/bin/activate && python src/scripts/batch_inference.py --model_path '$MODEL_PATH' --folder '$INPUT_FOLDER' --prompt_type '$PROMPT_TYPE'"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output '$OUTPUT_FILE'"
fi

echo "Running command: $CMD"
echo ""

# Execute the command
eval $CMD

echo ""
echo "Batch inference completed!"
