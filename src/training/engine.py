import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def train_epoch(model, train_dataloader, optimizer, scheduler, loss_function, device, parent_child_pairs, H_LAMBDA):
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            
            main_loss = loss_function(logits, labels)
            
            probabilities = torch.sigmoid(logits)
            hierarchical_penalty = 0.0
            
            for sub_id, narr_id in parent_child_pairs:
                # Get the probabilities for the parent and child for the entire batch
                p_parent = probabilities[:, narr_id]
                p_child = probabilities[:, sub_id]
                penalty_for_pair = torch.mean(torch.clamp(p_child - p_parent, min=0))
                
                hierarchical_penalty += penalty_for_pair

            loss = main_loss + hierarchical_penalty * H_LAMBDA
            
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        return avg_train_loss
    
def evaluate(model, eval_dataloader, loss_function, device, H_LAMBDA, parent_child_pairs, threshold=0.5):
    
    print("Running evaluation on the validation set...")
    model.eval() # Set the model to evaluation mode
    
    
    total_val_loss = 0.0
    all_preds_logits = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Validating"):
            # Move batch data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass to get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            val_main_loss = loss_function(logits, labels)
            val_probs = torch.sigmoid(logits)
            val_hierarchical_penalty = 0.0
            for sub_id, narr_id in parent_child_pairs:
                p_parent = val_probs[:, narr_id]
                p_child = val_probs[:, sub_id]
                val_hierarchical_penalty += torch.mean(torch.clamp(p_child - p_parent, min=0))
                
            val_loss = val_main_loss + (H_LAMBDA * val_hierarchical_penalty)
            total_val_loss += val_loss.item()
            
            all_preds_logits.append(logits.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(eval_dataloader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    
    all_preds_logits = np.concatenate(all_preds_logits, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    metrics = compute_metrics(all_preds_logits, all_true_labels, parent_child_pairs, threshold)

    return avg_val_loss, metrics

def compute_metrics(all_preds_logits, all_true_labels, parent_child_pairs, threshold=0.5):
    # --- Metric Calculation ---
    # Convert logits to probabilities
    sigmoid_preds = 1 / (1 + np.exp(-all_preds_logits))
    # Convert probabilities to binary predictions based on the threshold
    binary_preds = (sigmoid_preds > threshold).astype(int)

    # --- Hierarchical Correction at Inference Time (Optional but Recommended) ---
    # This step ensures the final output is 100% consistent
    for sub_id, narr_id in parent_child_pairs:
        inconsistent_mask = (binary_preds[:, sub_id] == 1) & (binary_preds[:, narr_id] == 0)
        binary_preds[inconsistent_mask, sub_id] = 0

    # Calculate metrics on the (potentially corrected) binary predictions
    f1_micro = f1_score(y_true=all_true_labels, y_pred=binary_preds, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=all_true_labels, y_pred=binary_preds, average='macro', zero_division=0)
    
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }
    return metrics

def get_raw_predictions(model, dataloader, device):
    """
    Runs the model on a dataset and returns its raw logits and the true labels.
    This is used to get the data needed for threshold finding.
    """
    
    model.eval()

    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting Raw Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_logits, all_labels
