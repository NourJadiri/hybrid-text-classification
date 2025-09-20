import json
import os
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
ARTIFACTS_DIR = 'mdeberta_artifacts_hierarchical'
DATASET_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, 'tokenized_dataset_hierarchical')

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
print(f"Total annotations after filtering 'Other': {len(df_clean)}")

# --------------------------------------------------------------------------------------
print("Step 2: Discovering Hierarchical Labels")

all_parents = set()
hierarchy = {}

for _, row in df_clean.iterrows():
    parents_in_row = {p.strip() for p in row['narratives'].split(';')}
    all_parents.update(parents_in_row)
    
    subnarratives = row['subnarratives'].split(';')
    for sub in subnarratives:
        sub = sub.strip()
        
        parts = sub.rsplit(':', 1)
        
        if len(parts) == 2:
            parent, child = parts[0].strip(), sub
            
            if parent in hierarchy:
                hierarchy[parent].add(child)
            else:
                hierarchy[parent] = {child}
            all_parents.add(parent)
            
sorted_parents = sorted(list(all_parents))
parent_label2id = {label: i for i, label in enumerate(sorted_parents)}
parent_id2label = {i: label for i, label in enumerate(sorted_parents)}
   
child_label_maps = {}

for parent_name, children_set in hierarchy.items():
    # Add "Other" child for each parent
    other_child = f"{parent_name}: Other"
    children_set.add(other_child)
    
    sorted_children = sorted(list(children_set))
    label2id = {label: i for i, label in enumerate(sorted_children)}
    id2label = {i: label for i, label in enumerate(sorted_children)}
    child_label_maps[parent_name] = {
        'label2id': label2id,
        'id2label': id2label
    }
    
print(f"Discovered {len(parent_label2id)} unique parent narratives.")
print(f"Discovered child narratives for {len(child_label_maps)} of these parents.")
print("Example: Children for 'URW: Discrediting Ukraine':")
if 'URW: Discrediting Ukraine' in child_label_maps:
    print(child_label_maps['URW: Discrediting Ukraine']['label2id'].keys())

# --------------------------------------------------------------------------------------
# Step 3 : Assembling label rows
# --------------------------------------------------------------------------------------
def sanitize_name(name):
    """Creates a safe name for columns and dictionary keys."""
    import re
    name = re.sub(r'[^\w\s]', '', name) # Remove punctuation
    name = re.sub(r'\s+', '_', name)    # Replace spaces with underscores
    return name.lower()

# Create mapping for label recovery - CRITICAL for recovering original names
sanitized_to_original = {}
for parent_name in child_label_maps.keys():
    sanitized_name = sanitize_name(parent_name)
    sanitized_to_original[sanitized_name] = parent_name

def recover_original_name(sanitized_name):
    """Recover the original human-readable name from a sanitized name."""
    return sanitized_to_original.get(sanitized_name, sanitized_name)

def get_child_labels_for_parent(sanitized_parent_name):
    """Get original child labels for a sanitized parent name."""
    original_parent = recover_original_name(sanitized_parent_name)
    if original_parent in child_label_maps:
        return list(child_label_maps[original_parent]['label2id'].keys())
    return []
    
    
print("Step 3: Assembling label rows")

data_rows = []

label_columns = ['parent_labels']
for parent_name in child_label_maps.keys():
    label_columns.append(f"child_labels_{sanitize_name(parent_name)}")
    
for _, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
    label_vectors = {}
    
    parent_vector = np.zeros(len(parent_label2id), dtype=int)
    
    for narrative in row['narratives'].split(';'):
        narrative = narrative.strip()
        if narrative in parent_label2id:
            parent_vector[parent_label2id[narrative]] = 1
        
    label_vectors['parent_labels'] = parent_vector
    
    for parent_name, child_map_info in child_label_maps.items():
        num_children = len(child_map_info['label2id'])
        child_vector = np.zeros(num_children, dtype=int)
        label_vectors[f"child_labels_{sanitize_name(parent_name)}"] = child_vector
        
    for sub in row['subnarratives'].split(';'):
        sub = sub.strip()
        
        parts = sub.rsplit(':', 1)
        if len(parts) == 2:
            parent, child = parts[0].strip(), sub
            if parent in child_label_maps and child in child_label_maps[parent]['label2id']:
                col_name = f"child_labels_{sanitize_name(parent)}"
                idx = child_label_maps[parent]['label2id'][child]
                label_vectors[col_name][idx] = 1
                
    file_path = os.path.join(TEXT_FILE_DIR, row['filename'])
    with open(file_path, 'r') as f:
        text = f.read().strip()
        
    data_rows.append({'text': text, **label_vectors})
    
print(f"Constructed {len(data_rows)} data rows with hierarchical labels.")
dataset = Dataset.from_list(data_rows)

print("Example data row:")
print(dataset[0])

print("Calculating positional weights for all heads")
pos_weights_dict = {}
temp_df = pd.DataFrame(data_rows)
epsilon = 1e-8

for col in label_columns:
    label_matrix = np.array(temp_df[col].tolist())
    
    if label_matrix.ndim == 1:
        label_matrix = label_matrix.reshape(-1, 1)
        
    positive_counts = np.sum(label_matrix, axis=0)
    negative_counts = label_matrix.shape[0] - positive_counts
    
    pos_weights = (negative_counts + epsilon) / (positive_counts + epsilon)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)
    pos_weights_dict[col] = pos_weights_tensor
    print(f"Positional weights for {col}: {pos_weights_tensor.numpy()}")
    

# --------------------------------------------------------------------------------------
print("Step 4: Tokenization and Dataset Preparation")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set format for PyTorch, including all our new label columns
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'] + label_columns)
print("Tokenization and formatting complete.")

final_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)

print("Processing Complete. Final dataset structure:")
print(final_dataset)
print("\nColumns in the dataset:", final_dataset['train'].column_names)

# --------------------------------------------------------------------------------------

print("\nStep 6: Saving the hierarchical dataset and artifacts to disk...")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
final_dataset.save_to_disk(DATASET_OUTPUT_PATH)

# Save the dictionary of weight tensors
torch.save(pos_weights_dict, os.path.join(ARTIFACTS_DIR, "pos_weights_hierarchical.pt"))

# Save the new hierarchical label mappings
with open(os.path.join(ARTIFACTS_DIR, 'label_mappings_hierarchical.json'), 'w') as f:
    json.dump({
            "parent_label2id": parent_label2id,
            "parent_id2label": parent_id2label,
            "child_label_maps": child_label_maps
        }, f, indent=4)

print("Hierarchical artifacts saved successfully.")