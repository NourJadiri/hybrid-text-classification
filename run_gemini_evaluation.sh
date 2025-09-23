#!/bin/bash

# Batch evaluation script for Gemini 2.5 Flash narrative validation devset predictions
echo "==============================================================================="
echo "RUNNING EVALUATION FOR GEMINI 2.5 FLASH NARRATIVE VALIDATION DEVSET PREDICTIONS"
echo "==============================================================================="

# Create output directory
OUTPUT_DIR="results/analysis/gemini25_flash_narr_val_evaluation"
mkdir -p "$OUTPUT_DIR"

# Define languages and their corresponding files
declare -A languages=(
    ["BG"]="bg_dev_results.txt"
    ["EN"]="en_dev_results.txt"
    ["HI"]="hi_dev_results.txt"
    ["PT"]="pt_dev_results.txt"
    ["RU"]="ru_dev_results.txt"
)

DEVSET_DIR="devset"
PREDICTIONS_DIR="results/devset/gemini25_flash_narr_val"
MODEL_NAME="Google Gemini 2.5 Flash"
EXPERIMENT_NAME="Gemini 2.5 Flash - Narrative Validation Devset"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Devset directory: $DEVSET_DIR"
echo "  Predictions directory: $PREDICTIONS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation for each language
for lang in "${!languages[@]}"; do
    echo "========================================================================"
    echo "Evaluating language: $lang"
    echo "========================================================================"
    
    ground_truth="$DEVSET_DIR/$lang/subtask-2-annotations.txt"
    predictions="$PREDICTIONS_DIR/${languages[$lang]}"
    
    echo "Ground truth: $ground_truth"
    echo "Predictions: $predictions"
    
    if [[ -f "$ground_truth" && -f "$predictions" ]]; then
        python src/analysis/evaluate_devset_predictions.py \
            --ground_truth "$ground_truth" \
            --predictions "$predictions" \
            --language "$lang" \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --experiment_name "$EXPERIMENT_NAME"
        
        echo "✓ Completed evaluation for $lang"
    else
        echo "✗ Missing files for $lang:"
        [[ ! -f "$ground_truth" ]] && echo "  - Ground truth: $ground_truth"
        [[ ! -f "$predictions" ]] && echo "  - Predictions: $predictions"
    fi
    echo ""
done

echo "========================================================================"
echo "ALL EVALUATIONS COMPLETED!"
echo "========================================================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"