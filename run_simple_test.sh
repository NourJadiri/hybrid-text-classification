#!/bin/bash
# run_simple_test.sh - Example usage of simple inference script

cd /home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification

# Example 1: Test with a sample text
python src/scripts/simple_inference.py \
    --text "The Western sanctions are only strengthening Russia's resolve and economy, while hurting European citizens. This is a strategic failure by NATO." \
    --use_simple_prompt

echo ""
echo "To test your own text, run:"
echo "python src/scripts/simple_inference.py --text 'YOUR_TEXT_HERE' --use_simple_prompt"
