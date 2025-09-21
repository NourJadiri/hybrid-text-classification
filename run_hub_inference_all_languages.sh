#!/bin/bash

# Script to run inference on all languages using the HuggingFace Hub model
# AWCO/mdeberta-v3-base-narratives-classifier-hierarchical

echo "Running inference on all languages in devset using HuggingFace Hub model..."
echo "Model: AWCO/mdeberta-v3-base-narratives-classifier-hierarchical"
echo ""

# Change to the mDeberta source directory
cd "$(dirname "$0")"

# Create results directory if it doesn't exist
mkdir -p ../../results/hub_inference

# Run inference on all languages
python run_inference_hub.py \
    --model_name "AWCO/mdeberta-v3-base-narratives-classifier-hierarchical" \
    --devset_dir "../../devset" \
    --output_dir "../../results/hub_inference" \
    --threshold 0.5 \
    --all_languages

echo ""
echo "Inference completed! Results saved in ../../results/hub_inference/"
echo "Output files:"
echo "- BG_predictions.tsv"
echo "- EN_predictions.tsv" 
echo "- HI_predictions.tsv"
echo "- PT_predictions.tsv"
echo "- RU_predictions.tsv"