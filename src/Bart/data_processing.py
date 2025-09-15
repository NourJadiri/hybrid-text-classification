import pandas as pd
import numpy as np
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# --- Configuration ---
# Updated paths to match the actual data structure
DATA_DIR = 'data/subtask-2-translated/'
ANNOTATION_FILE = 'data/subtask-2-translated/subtask-2-annotations.txt'
MODEL_NAME = 'allenai/longformer-base-4096'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

# --- Sub-step 2.1: Discover All Unique Narrative and Sub-narrative Labels ---

print("--- Step 2.1: Discovering All Unique Labels from the Dataset ---")

# Load annotations from the consolidated annotation file
print(f"Loading annotations from {ANNOTATION_FILE}")
if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_FILE}")

df_for_discovery = pd.read_csv(
    ANNOTATION_FILE,
    sep='\t',
    header=None,
    names=['filename', 'narratives', 'subnarratives']  # Based on the actual format
)

unique_labels = set()

# Gather all unique labels from both the narrative and sub-narrative columns
for _, row in df_for_discovery.iterrows():
    # Process Narratives (L1)
    if pd.notna(row['narratives']):
        labels = {lbl.strip() for lbl in row['narratives'].split(';')}
        unique_labels.update(labels)
        
    # Process Sub-narratives (L2)
    if pd.notna(row['subnarratives']):
        labels = {lbl.strip() for lbl in row['subnarratives'].split(';')}
        unique_labels.update(labels)

# Remove the generic 'Other' label if it exists
if 'Other' in unique_labels:
    unique_labels.remove('Other')

# Create the final sorted list and mappings
sorted_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(sorted_labels)}
id2label = {i: label for i, label in enumerate(sorted_labels)}
num_labels = len(sorted_labels)

print(f"Discovered {num_labels} unique labels (Narratives and Sub-narratives).")
print("Example labels:", json.dumps(sorted_labels[:5], indent=2))

# --- Sub-step 2.2: Generate Semantic Embeddings for Labels ---

print(f"\n--- Step 2.2: Generating Label Embeddings using '{EMBEDDING_MODEL_NAME}' ---")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Encoding labels into semantic vectors...")
label_embeddings_np = embedding_model.encode(sorted_labels)
label_embeddings = torch.tensor(label_embeddings_np, dtype=torch.float)
print(f"Label embeddings created with shape: {label_embeddings.shape}")

# --- Sub-step 2.3: Load Texts and Create Final Label Vectors ---

print("\n--- Step 2.3: Loading Texts and Preparing Data Rows ---")
data_rows = []
for _, row in tqdm(df_for_discovery.iterrows(), total=len(df_for_discovery), desc="Processing rows"):
    if row['narratives'] == 'Other':
        continue
    
    # Build correct file path - all files are in the subtask-2-translated folder
    filename = row['filename']
    file_path = os.path.join(DATA_DIR, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Warning: File not found, skipping: {file_path}")
        continue

    active_labels = set()
    if pd.notna(row['narratives']):
        active_labels.update({lbl.strip() for lbl in row['narratives'].split(';')})
    if pd.notna(row['subnarratives']):
        active_labels.update({lbl.strip() for lbl in row['subnarratives'].split(';')})

    label_vector = [0] * num_labels
    for label_str in active_labels:
        if label_str in label2id:
            label_id = label2id[label_str]
            label_vector[label_id] = 1

    data_rows.append({"text": text, "labels": label_vector})

processed_df = pd.DataFrame(data_rows)
print(f"\nProcessed {len(processed_df)} documents.")
print("Sample processed row:")
print(processed_df.iloc[0].to_dict())

# --- Sub-step 2.4: Tokenizing with Longformer and Creating Final Dataset ---

print(f"\n--- Step 2.4: Tokenizing with '{MODEL_NAME}' Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_dataset = Dataset.from_pandas(processed_df)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=4096
    )

print("Tokenizing the dataset...")
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

final_datasets = tokenized_dataset.train_test_split(test_size=0.15, seed=42)

print("\nPreprocessing complete!")
print("Final dataset structure:")
print(final_datasets)

# --- Save artifacts for the next step ---
print("\nSaving artifacts for training...")
final_datasets.save_to_disk("data/processed/tokenized_hierarchical_dataset")
torch.save(label_embeddings, "embeddings/label_embeddings.pt")
with open("label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)

print("\n✅ All necessary files are saved and ready for Step 3: Training.")