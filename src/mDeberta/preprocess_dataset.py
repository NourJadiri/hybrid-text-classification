import os
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Configuration
ANNOTATION_FILE = 'data/all-texts-unified/unified-annotations.tsv'
TEXT_FILE_DIR = 'data/all-texts-unified/texts'
BASE_MODEL = 'microsoft/mdeberta-v3-base'
MAX_LENGTH = 512
ARTIFACTS_DIR = 'mdeberta_artifacts'
DATASET_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, 'tokenized_dataset')

# --------------------------------------------------------------------------------------
# Step 1.1: Loading and Filtering Annotations
# --------------------------------------------------------------------------------------
print("Step 1.1: Loading and Filtering Annotations")

df_all = pd.read_csv(
    ANNOTATION_FILE,
    sep='\t',
    header=None,
    names=['filename', 'narratives', 'subnarratives']
)

# Keep consistent with previous script: filter out 'Other'
df_clean = df_all[df_all['narratives'] != 'Other'].reset_index(drop=True)
print(f"Loaded and filtered {len(df_clean)} labeled documents.")

# --------------------------------------------------------------------------------------
# Step 1.2: Discovering Unique Narratives (labels)
# --------------------------------------------------------------------------------------
print("\nStep 1.2: Discovering Unique Narratives")

unique_labels = set()
for narratives in df_clean['narratives']:
    unique_labels.update({narrative.strip() for narrative in narratives.split(';')})

sorted_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(sorted_labels)}
id2label = {i: label for i, label in enumerate(sorted_labels)}
num_labels = len(sorted_labels)
print(f"Discovered {num_labels} unique narratives.")

# --------------------------------------------------------------------------------------
# Step 1.3: Assembling data rows (vanilla MLC, no embeddings)
# --------------------------------------------------------------------------------------
print("\nStep 1.3: Assembling data rows")

data_rows = []
for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Assembling data rows"):
    narratives_str = row['narratives']
    filename = row['filename']
    file_path = os.path.join(TEXT_FILE_DIR, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    label_vector = np.zeros(num_labels, dtype=int)
    for narrative in (n.strip() for n in narratives_str.split(';')):
        if narrative in label2id:
            label_vector[label2id[narrative]] = 1

    data_rows.append({
        'text': text,
        'labels': label_vector
    })

print(f"Assembled {len(data_rows)} data rows.")

dataset = Dataset.from_list(data_rows)

# --------------------------------------------------------------------------------------
# Step 1.4: Calculating class weights for balancing
# --------------------------------------------------------------------------------------
print("\nStep 1.4: Calculating class weights for balancing...")

temp_df = pd.DataFrame(data_rows)
labels_matrix = np.array(temp_df['labels'].tolist())
positive_counts = labels_matrix.sum(axis=0)
negative_counts = len(temp_df) - positive_counts

# Avoid division by zero
epsilon = 1e-8
pos_weights = negative_counts / (positive_counts + epsilon)
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)

print(f"Calculated positive weights for {len(pos_weights_tensor)} labels.")
print("Example weights (first 10):", pos_weights_tensor[:10])

# --------------------------------------------------------------------------------------
# Step 1.5: Tokenizing the Dataset (mdeberta-v3-base)
# --------------------------------------------------------------------------------------
print("\nStep 1.5: Tokenizing the Dataset")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print("Tokenization complete.")

# Keep the same split pattern
final_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)

print("Processing Complete. Final dataset structure:")
print(final_dataset)

# --------------------------------------------------------------------------------------
# Saving artifacts
# --------------------------------------------------------------------------------------
print("Saving the tokenized dataset and artifacts to disk...")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
final_dataset.save_to_disk(DATASET_OUTPUT_PATH)

torch.save(pos_weights_tensor, os.path.join(ARTIFACTS_DIR, "pos_weights.pt"))

with open(os.path.join(ARTIFACTS_DIR, 'label_mappings.json'), 'w') as f:
    json.dump({
            "label2id": label2id,
            "id2label": id2label
        }, f, indent=4)

print("Artifacts saved successfully.")
