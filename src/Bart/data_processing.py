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
DATA_DIR = 'data/subtask-2-translated/'
ANNOTATION_FILE = 'data/subtask-2-translated/subtask-2-annotations.txt'
MODEL_NAME = 'allenai/longformer-base-4096'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # The 768-dim model

# --- Step 2.1: Load and Filter Annotations ONCE ---
print("--- Step 2.1: Loading and Filtering Annotations ---")
if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_FILE}")

# Load the complete annotations file
df_all = pd.read_csv(
    ANNOTATION_FILE,
    sep='\t',
    header=None,
    names=['filename', 'narratives', 'subnarratives']
)

# ** THE CRITICAL FIX: Filter the dataframe first and create our clean, final version **
df_clean = df_all[df_all['narratives'] != 'Other'].reset_index(drop=True)
print(f"Loaded and filtered {len(df_clean)} labeled documents.")

# --- Step 2.2: Discover Unique Labels from the CLEAN DataFrame ---
print("\n--- Step 2.2: Discovering Unique Labels ---")
unique_labels = set()
# ** Use the clean dataframe for ALL subsequent operations **
for _, row in df_clean.iterrows():
    if pd.notna(row['narratives']):
        unique_labels.update({lbl.strip() for lbl in row['narratives'].split(';')})
    if pd.notna(row['subnarratives']):
        unique_labels.update({lbl.strip() for lbl in row['subnarratives'].split(';')})

sorted_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(sorted_labels)}
id2label = {i: label for i, label in enumerate(sorted_labels)}
num_labels = len(sorted_labels)
print(f"Discovered {num_labels} unique labels (Narratives and Sub-narratives).")
print("Example labels:", json.dumps(sorted_labels[:5], indent=2))


# --- Step 2.3: Generate Semantic Embeddings for Labels ---
print(f"\n--- Step 2.3: Generating Label Embeddings using '{EMBEDDING_MODEL_NAME}' ---")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
label_embeddings_np = embedding_model.encode(sorted_labels)
label_embeddings = torch.tensor(label_embeddings_np, dtype=torch.float)
print(f"Label embeddings created with shape: {label_embeddings.shape}")


# --- Step 2.4: Assemble Data Rows from the CLEAN DataFrame ---
print("\n--- Step 2.4: Assembling Data Rows ---")
data_rows = []
missing_files = []
# ** Use the clean dataframe again, ensuring perfect alignment **
for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing rows"):
    filename = row['filename']
    file_path = os.path.join(DATA_DIR, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        missing_files.append({"index": idx, "filename": filename})
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

# Verify alignment
if missing_files:
    print(f"WARNING: {len(missing_files)} files were missing and skipped!")
    print("This could cause text-label misalignment in train/test split.")
    for missing in missing_files[:5]:  # Show first 5
        print(f"  Missing: {missing['filename']} (index {missing['index']})")
else:
    print("✅ All files found - perfect text-label alignment guaranteed!")

processed_df = pd.DataFrame(data_rows) # This df contains the 'labels' column with vectors

# --- NEW: Step 2.5: Calculate Class Weights for Imbalanced Data ---
print("\n--- Step 2.5: Calculating weights for imbalanced classes ---")

# Sum the label vectors across all documents to get counts for each label
labels_matrix = np.array(processed_df['labels'].tolist())
positive_counts = labels_matrix.sum(axis=0)
negative_counts = len(processed_df) - positive_counts

# Calculate pos_weight for each class. Add a small epsilon to avoid division by zero.
epsilon = 1e-8
pos_weights = negative_counts / (positive_counts + epsilon)

# Convert to a PyTorch tensor
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)

print(f"Calculated positive weights for {len(pos_weights_tensor)} labels.")
print("Example weights (first 10):", pos_weights_tensor[:10])

print(f"\nProcessed {len(processed_df)} documents.")
print(f"\n--- Step 2.6: Tokenizing with '{MODEL_NAME}' Tokenizer ---")
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
# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

final_datasets.save_to_disk("data/processed/tokenized_hierarchical_dataset")
torch.save(label_embeddings, "embeddings/label_embeddings.pt")
torch.save(pos_weights_tensor, "embeddings/pos_weights.pt")
with open("label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)

print("\n✅ All necessary files are saved and ready for Step 3: Training.")