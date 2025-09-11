from sklearn.metrics import f1_score
import torch
import numpy as np
from tqdm.auto import tqdm

def get_validation_predictions(model, dataloader, device):
    """
    Runs the model on the validation set and returns raw probabilities and true labels.
    """
    model.eval() # Set model to evaluation mode
    
    all_probabilities = []
    all_true_labels = []
    
    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Getting Validation Predictions"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass to get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Convert logits to probabilities using sigmoid
            probabilities = torch.sigmoid(logits)
            
            # Move probabilities and labels to CPU and convert to numpy arrays to accumulate them
            all_probabilities.append(probabilities.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())
            
    # Concatenate all predictions and labels from all batches
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    
    return all_probabilities, all_true_labels

def find_best_threshold(probabilities, true_labels, metric='f1_micro'):

    print(f"Searching for best threshold to optimize {metric}...")
    
    best_threshold = 0
    best_f1_score = 0
    
    # Iterate through a range of possible thresholds
    for threshold in tqdm(np.arange(0.1, 0.9, 0.01), desc="Searching Thresholds"):
        # Apply the current threshold to get binary predictions
        binary_preds = (probabilities > threshold).astype(int)
        
        # Calculate the F1 score for this threshold
        if metric == 'f1_micro':
            current_f1 = f1_score(y_true=true_labels, y_pred=binary_preds, average='micro', zero_division=0)
        elif metric == 'f1_macro':
            current_f1 = f1_score(y_true=true_labels, y_pred=binary_preds, average='macro', zero_division=0)
        else:
            raise ValueError("Metric must be 'f1_micro' or 'f1_macro'")
            
        # If this threshold gives a better F1 score, update our best values
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = threshold
            
    print(f"Search complete. Best Threshold: {best_threshold:.2f}, Best {metric}: {best_f1_score:.4f}")
    return best_threshold, best_f1_score

def compute_f1_metrics(preds, labels, threshold=0.5):
    # preds are the raw logits, labels are the multi-hot encoded true values
    sigmoid_preds = 1 / (1 + np.exp(-preds)) # Apply sigmoid to convert logits to probabilities
    binary_preds = (sigmoid_preds > threshold).astype(int) # Apply threshold to get binary predictions
    
    f1_micro = f1_score(y_true=labels, y_pred=binary_preds, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=labels, y_pred=binary_preds, average='macro', zero_division=0)

    return {"f1_micro": f1_micro, "f1_macro": f1_macro}