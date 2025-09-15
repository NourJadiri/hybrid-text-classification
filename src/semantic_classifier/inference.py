import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from safetensors.torch import load_file
from model import NarrativesClassifier

BASE_MODEL = 'allenai/longformer-base-4096'
ARTIFACTS_PATH = 'narrative_model_artifacts/'
FINAL_MODEL_PATH = 'models/longformer_narratives_classifier'
PREDICTION_THRESHOLD = 0.5
TEST_DATA_DIR = 'testset/EN/subtask-2-documents'
OUTPUT_FILE_PATH = 'results/longformer/narrative_predictions.txt'

print("Loading the trained model and tokenizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

label_embeddings = torch.load(os.path.join(ARTIFACTS_PATH, 'narrative_embeddings.pt'))
pos_weights = torch.load(os.path.join(ARTIFACTS_PATH, 'pos_weights.pt'))

with open(os.path.join(ARTIFACTS_PATH, 'label_mappings.json'), 'r') as f:
    label_mappings = json.load(f)
id2label = label_mappings['id2label']
num_labels = len(id2label)
print(f"Number of labels: {num_labels}")

model = NarrativesClassifier(
    model_name=BASE_MODEL,
    label_embeddings=label_embeddings,
    pos_weights=pos_weights,
)

weights_path = os.path.join(FINAL_MODEL_PATH, 'model.safetensors')
state_dict = load_file(weights_path, device="cpu")
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")


def get_prediction_for_text(text):
    inputs = tokenizer(
        text,
        max_length=2048,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    predicted_labels = [id2label[i] for i, prob in enumerate(probs) if prob >= PREDICTION_THRESHOLD]
    return predicted_labels, probs

print(f"\n--- Processing files in '{TEST_DATA_DIR}' ---")

test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.txt') and not f.startswith('subtask-2-annotations')]
print(f"Found {len(test_files)} text files for prediction.")

results_for_df = []

for filename in tqdm(test_files, desc="Processing test files"):
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    predicted_labels, probs = get_prediction_for_text(text)
    results_for_df.append({
        'filename': filename,
        'predicted_narratives': ';'.join(predicted_labels) if predicted_labels else 'None',
        'probabilities': probs.tolist()
    })
    
print(f"\nWriting predictions to '{OUTPUT_FILE_PATH}'...")
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as out_f:
    out_f.write("filename\tpredicted_narratives\n")
    for result in results_for_df:
        out_f.write(f"{result['filename']}\t{result['predicted_narratives']}\n")
print("Prediction process completed.")
