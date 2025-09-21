#!/bin/bash

# Script to run performance analysis on the devset predictions

echo "Running performance analysis on devset predictions..."
echo "Threshold: 0.65"
echo ""

# Change to the analysis source directory
cd "$(dirname "$0")/src/analysis"

# Create analysis results directory if it doesn't exist
mkdir -p ../../results/analysis

# Run the analysis
python performance_analysis.py \
    --devset_dir "../../devset" \
    --predictions_dir "../../results/hub_inference" \
    --threshold_folder "0.65" \
    --output_dir "../../results/analysis" \
    --languages BG EN HI PT RU

echo ""
echo "Analysis completed! Results saved in ../../results/analysis/0.65/"
echo "Files generated:"
echo "- detailed_results.json (detailed per-language results)"
echo "- performance_summary.csv (summary table)"
echo "- plots/ (performance visualization plots)"