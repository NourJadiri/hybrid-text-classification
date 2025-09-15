import pandas as pd
import numpy as np
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

ANNOTATION_FILE = 'data/subtask-2-translated/subtask-2-annotations.txt'
BASE_MODEL = 'allenai/longformer-base-4096'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
TEXT_FILE_DIR = 'data/subtask-2-translated/'
DATASET_OUTPUT_PATH = 'narrative_model_artifacts/tokenized_dataset'

print("Step 1.1: Loading and Filtering Annotations")

df_all = pd.read_csv(
    ANNOTATION_FILE,
    sep='\t',
    header=None,
    names=['filename', 'narratives', 'subnarratives']
)

df_clean = df_all[df_all['narratives'] != 'Other'].reset_index(drop=True)
print(f"Loaded and filtered {len(df_clean)} labeled documents.")

print("\nStep 1.2: Discovering Unique narratives")

unique_labels = set()

for narratives in df_clean['narratives']:
    unique_labels.update({narrative.strip() for narrative in narratives.split(';')})
    
sorted_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(sorted_labels)}
id2label = {i: label for i, label in enumerate(sorted_labels)}
num_labels = len(sorted_labels)
print(f"Discovered {num_labels} unique narratives.")

print("\nStep 1.3: Generating Narrative Embeddings")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
narrative_embeddings_np = embedding_model.encode(sorted_labels)
narrative_embeddings = torch.tensor(narrative_embeddings_np, dtype=torch.float)
print(f"Narrative embeddings created with shape: {narrative_embeddings.shape}")

data_rows = []

for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc = "Assembling data rows"):
    narratives_str = row['narratives']
    filename = row['filename']
    file_path = os.path.join(TEXT_FILE_DIR, filename)
    narratives = [narrative.strip() for narrative in narratives_str.split(';')]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    label_vector = np.zeros(num_labels, dtype=int)
    
    for narrative in narratives:
        if narrative in label2id:
            label_index = label2id[narrative]
            label_vector[label_index] = 1

    data_rows.append({
        'text': text,
        'labels': label_vector
    })

print(f"Assembled {len(data_rows)} data rows.")
dataset = Dataset.from_list(data_rows)


print("\nStep 1.4: Tokenizing the Dataset")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print("Tokenization complete.")

final_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)

print("Processing Complete. Final dataset structure:")
print(final_dataset)

print("Saving the tokenized dataset to disk...")

os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
final_dataset.save_to_disk(DATASET_OUTPUT_PATH)
torch.save(narrative_embeddings, 'narrative_model_artifacts/narrative_embeddings.pt')

with open('narrative_model_artifacts/label_mappings.json', 'w') as f:
    json.dump({
            "label2id": label2id,
            "id2label": id2label
        }, 
        f, 
        indent=4
    )
    
print("Artifacts saved successfully.")