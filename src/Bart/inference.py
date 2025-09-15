import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import pandas as pd
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm.auto import tqdm

# --- Configuration ---
MODEL_PATH = "./final_semantic_classifier"
LABEL_EMBEDDINGS_PATH = "embeddings/label_embeddings.pt"
LABEL_MAPPINGS_PATH = "label_mappings.json"
TEST_DATA_DIR = "testset/EN/subtask-2-documents"
PREDICTION_THRESHOLD = 0.43
# New: Path for the output file
OUTPUT_FILE_PATH = "predictions.txt"
POS_WEIGHTS_PATH = "embeddings/pos_weights.pt"
from Bart.training import POS_WEIGHTS_PATH
from semantic_classifier import SemanticClassifier

# --- Step 2: Load Model, Tokenizer, and Label Information ---
print("--- Loading model, tokenizer, and label info ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
label_embeddings = torch.load(LABEL_EMBEDDINGS_PATH)
pos_weights = torch.load(POS_WEIGHTS_PATH)
with open(LABEL_MAPPINGS_PATH, 'r') as f:
    label_mappings = json.load(f)
id2label = {int(k): v for k, v in label_mappings['id2label'].items()}
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = SemanticClassifier(model_name=MODEL_PATH, label_embeddings=label_embeddings, pos_weights=pos_weights)
model.to(device)
model.eval()
print("✅ Model and tokenizer loaded successfully.")

# --- Step 3: Define the Prediction Function ---
def predict(text: str):
    # (This function is unchanged, but now returns the full list of labels)
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=4096, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
    predictions = (probs >= PREDICTION_THRESHOLD).astype(int)
    predicted_labels = [id2label[i] for i, val in enumerate(predictions) if val == 1]
    return predicted_labels

# --- Step 4: Run Inference and Collect Results ---
print(f"\n--- Running inference on files in '{TEST_DATA_DIR}' ---")
try:
    test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.txt')]
    if not test_files:
        print(f"Warning: No .txt files found in '{TEST_DATA_DIR}'")
except FileNotFoundError:
    print(f"Error: Directory not found: '{TEST_DATA_DIR}'")
    test_files = []

# New: A list to store our results as dictionaries
all_results = []

for filename in tqdm(test_files, desc="Classifying documents"):
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    predicted_labels = predict(content)
    
    # New: Separate predicted labels into Narratives (L1) and Sub-narratives (L2)
    predicted_narratives = []
    predicted_subnarratives = []
    
    for label in predicted_labels:
        # A simple way to distinguish is by the presence of a colon ':'
        # This assumes L1 labels do not contain colons, and L2 labels do.
        # Adjust this logic if your label format is different.
        if ":" in label:
            # This is likely a Sub-narrative, but we need to be careful.
            # A better check: Does the label string match a known L1 format?
            # Let's check if it has a child part.
            parts = label.split(':', 1)
            # If the part after the colon is not empty, it's a sub-narrative.
            if len(parts) > 1 and parts[1].strip():
                 predicted_subnarratives.append(label)
            else:
                 # This handles cases like "CC: Amplifying Climate Fears" which might have a colon but are L1
                 predicted_narratives.append(label)
        else:
            predicted_narratives.append(label)

    # Store the results for this file
    all_results.append({
        'filename': filename,
        # Join the lists with semicolons to match the original format
        'narratives': ";".join(sorted(predicted_narratives)),
        'subnarratives': ";".join(sorted(predicted_subnarratives))
    })

# --- Step 5: Write Results to TSV File ---
print(f"\n--- Writing {len(all_results)} predictions to '{OUTPUT_FILE_PATH}' ---")

if all_results:
    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Ensure the columns are in the correct order
    results_df = results_df[['filename', 'narratives', 'subnarratives']]
    
    # Write to a TSV file without the header and index
    results_df.to_csv(OUTPUT_FILE_PATH, sep='\t', header=False, index=False)
    
    print("✅ Predictions successfully written.")
else:
    print("No results to write.")