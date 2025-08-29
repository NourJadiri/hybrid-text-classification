#!/bin/bash
# test_single_inference.sh - Quick test script for single text inference

# Set the working directory to the project root
cd /home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification

# Test text (you can modify this)
TEST_TEXT="The Western sanctions are only strengthening Russia's resolve and economy, while hurting European citizens. This is a strategic failure by NATO that benefits Putin's regime."

# Expected narratives (optional - you can modify or remove this)
EXPECTED="URW: Russia is the Victim;URW: Discrediting the West"

# Run the inference test
python src/scripts/test_qwen_inference.py \
    --model_path "models/qwen/smoke-test-qwen-1.7b/final_model" \
    --base_model "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit" \
    --test_text "$TEST_TEXT" \
    --expected_narratives "$EXPECTED" \
    --output_path "outputs/single_test_result.json" \
    --use_simple_prompt

echo "Test completed! Check outputs/single_test_result.json for detailed results."
