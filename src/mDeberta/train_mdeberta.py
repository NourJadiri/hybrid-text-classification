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
# Metrics: compute sample-, micro-, and macro-F1 (primary: sample-F1)
# -----------------------------------------------------------------------------

def compute_metric(p):
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
# Custom Trainer to use class-wise pos_weight in BCEWithLogitsLoss
# -----------------------------------------------------------------------------
class WeightedBCELossTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional num_items_in_batch parameter for compatibility.
        The num_items_in_batch parameter is used by newer Transformers versions
        for accumulation handling but we don't need it for our BCE loss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Move pos_weight to the same device as logits if provided
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
        else:
            pos_weight = None
            
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss


def main():
    print("Loading preprocessed dataset and artifacts...")
    tokenized_datasets = load_from_disk(os.path.join(ARTIFACTS_PATH, 'tokenized_dataset'))
    pos_weights = torch.load(os.path.join(ARTIFACTS_PATH, 'pos_weights.pt'))

    with open(os.path.join(ARTIFACTS_PATH, 'label_mappings.json'), 'r') as f:
        label_mappings = json.load(f)
    # normalize JSON keys -> ints for HF config
    id2label = {int(k): v for k, v in label_mappings['id2label'].items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    num_labels = len(id2label)

    print(tokenized_datasets)

    print("Instantiating the model...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

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
        gradient_accumulation_steps=2,  # Effectively batch size of 8
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_sample",  # primary metric is F1 (sample)
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        warmup_ratio=0.1,  # Add warmup for stability
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,  # Speed up data loading
        remove_unused_columns=False,  # Important for custom loss
    )

    print("Creating the trainer...")
    trainer = WeightedBCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],  # type: ignore[arg-type]
        eval_dataset=tokenized_datasets['test'],    # type: ignore[arg-type]
        compute_metrics=compute_metric,
        pos_weight=pos_weights,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete! Saving the final model...")
    trainer.save_model(FINAL_MODEL_PATH)
    model.config.save_pretrained(FINAL_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)

    print("Model saved to", FINAL_MODEL_PATH)
    print("All done!")


if __name__ == "__main__":
    main()
