# First, ensure you have the necessary libraries installed:
# pip install torch transformers datasets scikit-learn accelerate

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datasets import load_from_disk
from transformers import AutoModel, AutoConfig, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from semantic_classifier import SemanticClassifier

# --- Configuration ---
MODEL_NAME = 'allenai/longformer-base-4096'
DATASET_PATH = "data/processed/tokenized_hierarchical_dataset"
LABEL_EMBEDDINGS_PATH = "embeddings/label_embeddings.pt"
LABEL_MAPPINGS_PATH = "label_mappings.json"
POS_WEIGHTS_PATH = "embeddings/pos_weights.pt"

# --- Step 3.2: Load All Preprocessed Artifacts ---

print("--- Step 3.2: Loading Preprocessed Data and Embeddings ---")
# Load the tokenized dataset
tokenized_datasets = load_from_disk(DATASET_PATH)
# Load the label embeddings
label_embeddings = torch.load(LABEL_EMBEDDINGS_PATH)
pos_weights = torch.load(POS_WEIGHTS_PATH)
# Load the label mappings
with open(LABEL_MAPPINGS_PATH, 'r') as f:
    label_mappings = json.load(f)
id2label = label_mappings['id2label']
num_labels = len(id2label)

print("Loaded tokenized dataset:", tokenized_datasets)
print(f"Loaded label embeddings with shape: {label_embeddings.shape}")

# --- Step 3.3: Define Evaluation Metrics ---

print("\n--- Step 3.3: Defining Evaluation Metrics ---")
# This function will be called by the Trainer at each evaluation step.
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    
    # 1. Get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    # --- New Logic: Find the best threshold ---
    # We will search for the threshold that maximizes the F1 score
    # We check thresholds from 0.1 to 0.9 in steps of 0.01
    best_f1 = 0
    best_threshold = 0.5 # Default
    
    # Flatten arrays for micro-averaging
    y_true_flat = labels.flatten()
    y_probs_flat = probs.numpy().flatten()
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        # Apply threshold to get binary predictions
        y_pred_binary_flat = (y_probs_flat >= threshold).astype(int)
        
        # Calculate F1 score for this threshold
        current_f1 = f1_score(y_true=y_true_flat, y_pred=y_pred_binary_flat, average='micro')
        
        # If this F1 is the best so far, update our best values
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            
    # --- End of New Logic ---
    
    # Now, calculate all metrics using the BEST threshold we found
    y_pred_best_binary = (probs.numpy() >= best_threshold).astype(int)
    
    accuracy_at_best_f1 = accuracy_score(y_true=labels, y_pred=y_pred_best_binary)
    
    # ROC AUC is independent of the threshold, so we calculate it as before
    roc_auc_micro = roc_auc_score(y_true=labels, y_score=probs, average='micro')
    
    # We can also report the optimal threshold itself, which is very useful!
    metrics = {
        'f1_micro': best_f1,
        'roc_auc_micro': roc_auc_micro,
        'accuracy_at_best_f1': accuracy_at_best_f1,
        'optimal_threshold': best_threshold
    }
    
    return metrics

# --- Step 3.4: Set Up the Trainer ---

print("\n--- Step 3.4: Setting Up the Trainer ---")
# Instantiate our custom model
model = SemanticClassifier(
    model_name=MODEL_NAME,
    label_embeddings=label_embeddings,
    pos_weights=pos_weights
)

# Define the training arguments
# These control various aspects of the fine-tuning process
training_args = TrainingArguments(
    output_dir='./results',              # Directory to save model checkpoints
    num_train_epochs=5,                  # Total number of training epochs
    per_device_train_batch_size=4,       # Batch size for training (adjust based on your GPU VRAM)
    per_device_eval_batch_size=2,        # Batch size for evaluation
    gradient_accumulation_steps=4,       # Increase effective batch size to 2*4=8
    warmup_ratio=0.2,                    # Number of steps for the learning rate warmup
    weight_decay=0.01,                   # Strength of weight decay regularization
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=50,                    # Log metrics every 50 steps
    eval_strategy="epoch",         # Run evaluation at the end of each epoch
    save_strategy="epoch",               # Save a checkpoint at the end of each epoch
    load_best_model_at_end=True,         # Load the best model found during training at the end
    metric_for_best_model="f1_micro",    # Use f1_micro to determine the "best" model
    fp16=True,       # Use mixed-precision training for speedup (requires CUDA)
    learning_rate=5e-6,
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],  # type: ignore[arg-type]
    eval_dataset=tokenized_datasets['test'],    # type: ignore[arg-type]
    compute_metrics=compute_metrics,
)

# --- Step 3.5: Start Training ---

print("\n--- Step 3.5: Starting Model Training ---")
print("This will take a while depending on your dataset size and GPU.")

trainer.train()

print("\nTraining finished!")

# --- Save the final model and tokenizer ---
final_model_path = "./final_semantic_classifier"
trainer.save_model(final_model_path)
model.bert.config.save_pretrained(final_model_path)

# Save the tokenizer separately (it's not stored in dataset features)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(final_model_path)

print(f"\n✅ Final model and tokenizer saved to '{final_model_path}'")
print("You can now use this model for inference.")