import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = 'models/mdeberta_narratives_classifier'
ARTIFACTS_PATH = 'mdeberta_artifacts/'
TEST_TEXTS_DIR = 'testset/EN/subtask-2-documents'  # Path to test documents
OUTPUT_FILE = 'mdeberta_predictions.txt'
THRESHOLD = 0.5  # Decision threshold for multi-label classification
MAX_LENGTH = 512  # Same as training

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_model_and_artifacts():
    """Load the trained model, tokenizer, and label mappings"""
    print("Loading trained model and artifacts...")
    
    # Load label mappings
    with open(os.path.join(ARTIFACTS_PATH, 'label_mappings.json'), 'r') as f:
        label_mappings = json.load(f)
        label2id = label_mappings['label2id']
        id2label = label_mappings['id2label']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded with {len(id2label)} labels")
    return model, tokenizer, label2id, id2label

def predict_text(text, model, tokenizer, id2label, threshold=0.5):
    """Predict narratives for a single text"""
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Move to device if available
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply sigmoid and threshold
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).int()
    
    # Convert to label names
    predicted_labels = []
    for i, pred in enumerate(predictions[0]):
        if pred == 1:
            predicted_labels.append(id2label[str(i)])
    
    return predicted_labels, probabilities[0].cpu().tolist()

def process_test_files(test_dir, model, tokenizer, id2label, threshold=0.5):
    """Process all test files and generate predictions"""
    print(f"Processing test files from {test_dir}...")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Get all text files
    text_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
    text_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(text_files)} test files")
    
    results = []
    
    for filename in tqdm(text_files, desc="Processing files"):
        file_path = os.path.join(test_dir, filename)
        
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Skip empty files
            if not text:
                print(f"Warning: Empty file {filename}")
                results.append({
                    'filename': filename,
                    'narratives': ['Other'],
                    'probabilities': []
                })
                continue
            
            # Make prediction
            predicted_labels, probabilities = predict_text(
                text, model, tokenizer, id2label, threshold
            )
            
            # If no labels predicted, use "Other"
            if not predicted_labels:
                predicted_labels = ['Other']
            
            results.append({
                'filename': filename,
                'narratives': predicted_labels,
                'probabilities': probabilities
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                'filename': filename,
                'narratives': ['Other'],
                'probabilities': []
            })
    
    return results

def save_predictions_tsv_format(results, output_file):
    """Save predictions in TSV format with only narratives, subnarratives filled with 'Other'"""
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            filename = result['filename']
            narratives = result['narratives']
            
            # Join narratives with semicolons
            narratives_str = ';'.join(narratives)
            
            # Always use "Other" for subnarratives column as requested
            subnarratives_str = "Other"
            
            # Write in TSV format: filename\tnarratives\tsubnarratives
            f.write(f"{filename}\t{narratives_str}\t{subnarratives_str}\n")
    
    print(f"Predictions saved to {output_file}")
    print("Note: All subnarratives set to 'Other' as requested")

def save_detailed_predictions(results, output_file):
    """Save detailed predictions with probabilities"""
    detailed_output = output_file.replace('.txt', '_detailed.json')
    
    print(f"Saving detailed predictions to {detailed_output}...")
    
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed predictions saved to {detailed_output}")

def print_prediction_summary(results, id2label):
    """Print a summary of predictions"""
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    total_files = len(results)
    files_with_predictions = sum(1 for r in results if r['narratives'] != ['Other'])
    files_with_other = total_files - files_with_predictions
    
    print(f"Total files processed: {total_files}")
    print(f"Files with narrative predictions: {files_with_predictions}")
    print(f"Files classified as 'Other': {files_with_other}")
    
    # Count label frequencies
    label_counts = {}
    for result in results:
        for label in result['narratives']:
            if label != 'Other':
                label_counts[label] = label_counts.get(label, 0) + 1
    
    if label_counts:
        print("\nMost frequent predicted narratives:")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels[:10]:  # Show top 10
            print(f"  {label}: {count} files")
    
    print("="*60)

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    print("Starting mDeberta Inference Pipeline")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run the training script first to create the model.")
        return
    
    # Load model and artifacts
    try:
        model, tokenizer, label2id, id2label = load_model_and_artifacts()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Process test files
    try:
        results = process_test_files(
            TEST_TEXTS_DIR, model, tokenizer, id2label, THRESHOLD
        )
    except Exception as e:
        print(f"ERROR: Failed to process test files: {e}")
        return
    
    # Save predictions in TSV format
    save_predictions_tsv_format(results, OUTPUT_FILE)
    
    # Save detailed predictions with probabilities
    save_detailed_predictions(results, OUTPUT_FILE)
    
    # Print summary
    print_prediction_summary(results, id2label)
    
    print(f"\nInference completed successfully!")
    print(f"Main output: {OUTPUT_FILE}")
    print(f"Detailed output: {OUTPUT_FILE.replace('.txt', '_detailed.json')}")

if __name__ == "__main__":
    main()