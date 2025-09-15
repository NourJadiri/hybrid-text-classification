import torch
import numpy as np
import json
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
from model import NarrativesClassifier
from compute_metric import compute_metric

MODEL_NAME = 'allenai/longformer-base-4096'
ARTIFACTS_PATH = 'narrative_model_artifacts/'
FINAL_MODEL_PATH = 'models/longformer_narratives_classifier'


print("Loading preprocessed data and embeddings...")

tokenized_datasets = load_from_disk(os.path.join(ARTIFACTS_PATH, 'tokenized_dataset'))
label_embeddings = torch.load(os.path.join(ARTIFACTS_PATH, 'narrative_embeddings.pt'))
pos_weights = torch.load(os.path.join(ARTIFACTS_PATH, 'pos_weights.pt'))

with open(os.path.join(ARTIFACTS_PATH, 'label_mappings.json'), 'r') as f:
    label_mappings = json.load(f)
id2label = label_mappings['id2label']
num_labels = len(id2label)

print(tokenized_datasets)

print("Instantiating the model...")
model = NarrativesClassifier(
    model_name=MODEL_NAME,
    label_embeddings=label_embeddings,
    pos_weights=pos_weights,
)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=FINAL_MODEL_PATH,
    
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    
    gradient_checkpointing=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    
    metric_for_best_model="roc_auc_micro",
    greater_is_better=True,
    
    logging_strategy="steps",
    logging_steps=50,
)


print("Creating the trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],  # type: ignore[arg-type]
    eval_dataset=tokenized_datasets['test'],    # type: ignore[arg-type]
    compute_metrics=compute_metric
)

print("Starting training...")
trainer.train()

print("Training complete!")
print("Saving the final model...")
trainer.save_model(FINAL_MODEL_PATH)
model.bert.config.save_pretrained(FINAL_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print("Model saved to", FINAL_MODEL_PATH)
print("All done!")