import os
import json
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# Configuration (mirrors your prior patterns)
# -----------------------------------------------------------------------------
MODEL_NAME = 'microsoft/mdeberta-v3-base'
ARTIFACTS_PATH = 'mdeberta_artifacts/'
FINAL_MODEL_PATH = 'models/mdeberta_narratives_classifier'
THRESHOLD = 0.5  # decision threshold for multi-label F1


# -----------------------------------------------------------------------------
# Load preprocessed dataset and artifacts
# -----------------------------------------------------------------------------
print("Loading preprocessed dataset and artifacts...")

# Load the tokenized dataset
dataset = load_from_disk(os.path.join(ARTIFACTS_PATH, 'tokenized_dataset'))
print(f"Loaded dataset: {dataset}")

# Load class weights
pos_weights = torch.load(os.path.join(ARTIFACTS_PATH, 'pos_weights.pt'))
print(f"Loaded class weights with shape: {pos_weights.shape}")

# Load label mappings
with open(os.path.join(ARTIFACTS_PATH, 'label_mappings.json'), 'r') as f:
    label_mappings = json.load(f)
    label2id = label_mappings['label2id']
    id2label = label_mappings['id2label']

num_labels = len(label2id)
print(f"Number of labels: {num_labels}")


# -----------------------------------------------------------------------------
# Custom Weighted BCE Loss Trainer
# -----------------------------------------------------------------------------
class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights.to(self.args.device if hasattr(self.args, 'device') else 'cpu')
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Move pos_weights to the same device as logits
        pos_weights = self.pos_weights.to(logits.device)
        
        # Weighted BCE loss
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        loss = loss_fn(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Metrics: compute sample-, micro-, and macro-F1 (primary: sample-F1)
# -----------------------------------------------------------------------------
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    # probabilities and binary predictions
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs >= THRESHOLD).to(torch.int)

    y_true = torch.tensor(labels)
    y_pred = preds

    # Convert to numpy for sklearn
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()

    try:
        f1_micro = f1_score(yt, yp, average='micro', zero_division=0)
    except Exception:
        f1_micro = 0.0
    try:
        f1_macro = f1_score(yt, yp, average='macro', zero_division=0)
    except Exception:
        f1_macro = 0.0
    try:
        f1_sample = f1_score(yt, yp, average='samples', zero_division=0)
    except Exception:
        f1_sample = 0.0

    return {
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'f1_sample': float(f1_sample),  # primary metric
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
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    problem_type="multi_label_classification"
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config
)

print(f"Model loaded with {num_labels} output labels")


# -----------------------------------------------------------------------------
# Training Arguments
# -----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_sample",
    greater_is_better=True,
    save_total_limit=3,
    seed=42,
    dataloader_pin_memory=False,
)

print("Training arguments configured")


# -----------------------------------------------------------------------------
# Initialize Trainer
# -----------------------------------------------------------------------------
trainer = WeightedBCETrainer(
    pos_weights=pos_weights,
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

print("Trainer initialized with weighted BCE loss")


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("Starting training...")
train_result = trainer.train()

print("Training completed!")
print(f"Training loss: {train_result.training_loss:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
eval_result = trainer.evaluate()

print("\nEvaluation Results:")
for key, value in eval_result.items():
    print(f"{key}: {value:.4f}")


# -----------------------------------------------------------------------------
# Save the final model
# -----------------------------------------------------------------------------
print("\nSaving the final model...")
os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

# Save model and tokenizer
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

# Save additional artifacts
with open(os.path.join(FINAL_MODEL_PATH, 'training_config.json'), 'w') as f:
    json.dump({
        'model_name': MODEL_NAME,
        'num_labels': num_labels,
        'threshold': THRESHOLD,
        'max_length': 512,  # From preprocessing
        'training_args': training_args.to_dict(),
        'final_metrics': eval_result
    }, f, indent=4)

print(f"Model saved to {FINAL_MODEL_PATH}")
print("Training completed successfully!")

