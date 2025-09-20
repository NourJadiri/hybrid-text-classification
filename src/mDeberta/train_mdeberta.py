import os
import json
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score

from multihead_deberta import MultiHeadDebertaForHierarchicalClassification

# -----------------------------------------------------------------------------
# Configuration (mirrors your prior patterns)
# -----------------------------------------------------------------------------
MODEL_NAME = 'microsoft/mdeberta-v3-base'
ARTIFACTS_PATH = 'mdeberta_artifacts_hierarchical/'
FINAL_MODEL_PATH = f'models/{MODEL_NAME}_narratives_classifier_hierarchical'
THRESHOLD = 0.5

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
MAX_EPOCHS = 10

THRESHOLD_SEARCH_RANGE = (0.1, 0.9)
THRESHOLD_SEARCH_STEPS = 17


def sanitize_name(name):
    """Creates a safe name for dictionary keys. Must be IDENTICAL to the one in preprocessing."""
    import re
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.lower()

# -----------------------------------------------------------------------------
# Load preprocessed dataset and artifacts
# -----------------------------------------------------------------------------
print("Loading preprocessed dataset and artifacts...")

# Load the tokenized dataset
dataset = load_from_disk(os.path.join(ARTIFACTS_PATH, 'tokenized_dataset_hierarchical'))
print(f"Loaded dataset")



# Load class weights
pos_weights_dict = torch.load(os.path.join(ARTIFACTS_PATH, 'pos_weights_hierarchical.pt'))
print(f"Loaded class weights for {len(pos_weights_dict)} labels")


# Load label mappings
with open(os.path.join(ARTIFACTS_PATH, 'label_mappings_hierarchical.json'), 'r') as f:
    label_mappings = json.load(f)
    parent_label2id = label_mappings['parent_label2id']
    parent_id2label = {int(k): v for k, v in label_mappings['parent_id2label'].items()}
    
    child_label_maps = label_mappings['child_label_maps']

num_parent_labels = len(parent_label2id)
print(f"Number of parent labels: {num_parent_labels}")

# -----------------------------------------------------------------------------
# Threshold optimization and metrics computation
# -----------------------------------------------------------------------------
_BEST_THRESHOLD_TRACKER = {'threshold': THRESHOLD}

def find_optimal_threshold(y_true, y_probs, search_range=(0.1, 0.9), steps=17):
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    best_score = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        score = f1_score(y_true, y_pred, average='samples', zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def compute_hierarchical_metrics(p):
    # 1. Unpack predictions and labels from the model output
    parent_logits = p.predictions[0]
    child_logits_dict = p.predictions[1]
    
    label_ids_dict = dict(zip(label_columns, p.label_ids))
    true_parent_labels = label_ids_dict['parent_labels']
    
    # 2. Reconstruct a flat ground truth matrix (y_true) for all labels
    # First, create a final mapping for all unique labels (parents and children)
    flat_label2id = {}
    next_id = 0
    # Add parents
    for label in sorted(parent_label2id.keys()):
        if label not in flat_label2id: flat_label2id[label] = next_id; next_id += 1
    # Add children
    for parent, child_info in child_label_maps.items():
        for child in sorted(child_info['label2id'].keys()):
            full_child_label = f"{parent}:{child}"
            if full_child_label not in flat_label2id: flat_label2id[full_child_label] = next_id; next_id += 1
    
    num_total_labels = len(flat_label2id)
    num_samples = true_parent_labels.shape[0]
    y_true = np.zeros((num_samples, num_total_labels), dtype=int)

    # Populate y_true
    for i in range(num_samples):
        # Parents
        for pid, label in parent_id2label.items():
            if true_parent_labels[i, pid] == 1:
                y_true[i, flat_label2id[label]] = 1
        # Children
        for parent, child_info in child_label_maps.items():
            safe_key = sanitize_name(parent)
            child_label_key = f"child_labels_{safe_key}"
            true_child_labels = label_ids_dict[child_label_key]
            child_id2label = child_info['id2label']
            for cid_str, label in child_id2label.items():
                cid = int(cid_str)
                if true_child_labels[i, cid] == 1:
                    full_child_label = f"{parent}:{label}"
                    y_true[i, flat_label2id[full_child_label]] = 1

    # 3. Get parent probabilities and find an optimal threshold for them
    parent_probs = torch.sigmoid(torch.tensor(parent_logits)).numpy()
    # For simplicity, we optimize the threshold on parent predictions only,
    # as this is the entry point to the hierarchy.
    optimal_threshold, _ = find_optimal_threshold(true_parent_labels, parent_probs)
    _BEST_THRESHOLD_TRACKER['threshold'] = optimal_threshold
    
    # 4. Reconstruct a flat prediction matrix (y_pred) using the hierarchy
    y_pred = np.zeros((num_samples, num_total_labels), dtype=int)
    predicted_parents = (parent_probs >= optimal_threshold).astype(int)

    for i in range(num_samples):
        # Parents
        for pid, is_present in enumerate(predicted_parents[i]):
            if is_present:
                parent_name = parent_id2label[pid]
                y_pred[i, flat_label2id[parent_name]] = 1
                
                # Children (ONLY if parent is predicted)
                if parent_name in child_label_maps:
                    safe_key = sanitize_name(parent_name)
                    child_probs = torch.sigmoid(torch.tensor(child_logits_dict[safe_key][i])).numpy()
                    child_preds = (child_probs >= optimal_threshold).astype(int) # Use same threshold for simplicity
                    
                    child_id2label = child_label_maps[parent_name]['id2label']
                    for cid_str, child_name in child_id2label.items():
                        cid = int(cid_str)
                        if child_preds[cid] == 1:
                            full_child_label = f"{parent_name}:{child_name}"
                            y_pred[i, flat_label2id[full_child_label]] = 1

    # 5. Calculate final F1 scores on the flat matrices
    f1_sample = f1_score(y_true, y_pred, average='samples', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'f1_sample': f1_sample,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'optimal_threshold': optimal_threshold,
    }


# -----------------------------------------------------------------------------
# Model Setup and Configuration
# -----------------------------------------------------------------------------
print("Setting up model configuration...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure model for multi-label classification
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification"
)

# Load model
model = MultiHeadDebertaForHierarchicalClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    parent_label2id=parent_label2id,
    child_label_maps=child_label_maps,
    pos_weight_dict=pos_weights_dict
)

print(f"Model loaded with {num_parent_labels} output labels")


feature_columns = ['input_ids', 'attention_mask', 'token_type_ids'] 
# All other columns must be labels
label_columns = [col for col in dataset['train'].column_names if col not in feature_columns]

print(f"Identified label columns for the Trainer: {label_columns}")


training_args = TrainingArguments(
    output_dir='./results_hierarchical', # Use a new output dir
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=8,  # May need to reduce batch size for larger models
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2, # Compensate for smaller batch size
    warmup_ratio=0.1,
    logging_dir='./logs_hierarchical',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_sample", # This must match a key in compute_metrics output
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    dataloader_pin_memory=False,
    label_names = label_columns
)
print("Training arguments configured.")

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=EARLY_STOPPING_THRESHOLD
)

# We now use the standard Trainer class!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_hierarchical_metrics,
    callbacks=[early_stopping],
)

print("Standard Trainer initialized for hierarchical model.")


print("Starting training...")
train_result = trainer.train()

print("Training completed!")
print(f"Training loss: {train_result.training_loss:.4f}")

print("\nEvaluating on test set...")
eval_result = trainer.evaluate()

print("\nFinal Evaluation Results:")
for key, value in eval_result.items():
    print(f"{key}: {value:.4f}")

best_threshold = _BEST_THRESHOLD_TRACKER['threshold']
print(f"\nBest threshold found during final evaluation: {best_threshold:.3f}")


print("\nSaving the final model...")
os.makedirs(FINAL_MODEL_PATH, exist_ok=True)

# The default save_model will now save our custom model's architecture and weights
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

# Save the hierarchical mappings needed for inference
with open(os.path.join(FINAL_MODEL_PATH, 'label_mappings_hierarchical.json'), 'w') as f:
    json.dump(label_mappings, f, indent=4)

# Save the final training configuration and metrics
with open(os.path.join(FINAL_MODEL_PATH, 'training_config.json'), 'w') as f:
    json.dump({
        'model_name': MODEL_NAME,
        'best_threshold': best_threshold,
        'max_length': 512,
        'training_args': training_args.to_dict(),
        'final_metrics': eval_result
    }, f, indent=4)

print(f"Model and artifacts saved to {FINAL_MODEL_PATH}")
print("Training completed successfully!")