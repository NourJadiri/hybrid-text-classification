import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

def find_best_threshold(all_preds_logits, all_true_labels, parent_child_pairs, compute_metrics_fn, metric_to_optimize='f1_micro'):
    """
    Searches for the best prediction threshold by calling a compute_metrics function.
    """
    print(f"Searching for the best threshold to optimize {metric_to_optimize}...")
    
    best_threshold = 0.0
    best_score = 0.0
    
    for threshold in tqdm(np.arange(0.1, 0.91, 0.01), desc="Searching Thresholds"):
        # Use the provided compute_metrics function for the current threshold
        metrics = compute_metrics_fn(
            all_preds_logits, 
            all_true_labels, 
            parent_child_pairs, 
            threshold=threshold
        )
        current_score = metrics[metric_to_optimize]
            
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            
    print("Search complete!")
    print(f"Best Threshold found: {best_threshold:.2f}")
    print(f"Best Validation {metric_to_optimize.capitalize()}: {best_score:.4f}")
    
    return best_threshold

def find_per_level_thresholds(
    probabilities, 
    true_labels, 
    narrative_indices, 
    subnarrative_indices,
    parent_child_pairs
):
    """
    Finds the optimal separate thresholds for narrative and sub-narrative labels.
    """
    print("--- Searching for optimal NARRATIVE threshold (optimizing F1 Micro) ---")
    
    # Isolate the predictions and true labels for only the narrative columns
    narr_probs = probabilities[:, narrative_indices]
    narr_true = true_labels[:, narrative_indices]
    
    best_narr_threshold = 0.0
    best_narr_f1 = 0.0
    
    for threshold in tqdm(np.arange(0.1, 0.91, 0.01), desc="Narrative Thresholds"):
        binary_preds = (narr_probs > threshold).astype(int)
        current_f1 = f1_score(narr_true, binary_preds, average='micro', zero_division=0)
        if current_f1 > best_narr_f1:
            best_narr_f1 = current_f1
            best_narr_threshold = threshold
            
    print(f"Best Narrative Threshold: {best_narr_threshold:.2f} (F1 Micro: {best_narr_f1:.4f})")

    print("\n--- Searching for optimal SUB-NARRATIVE threshold (optimizing F1 Micro) ---")

    # Isolate the predictions and true labels for only the sub-narrative columns
    subnarr_probs = probabilities[:, subnarrative_indices]
    subnarr_true = true_labels[:, subnarrative_indices]

    best_subnarr_threshold = 0.0
    best_subnarr_f1 = 0.0

    for threshold in tqdm(np.arange(0.1, 0.91, 0.01), desc="Sub-narrative Thresholds"):
        binary_preds = (subnarr_probs > threshold).astype(int)
        current_f1 = f1_score(subnarr_true, binary_preds, average='micro', zero_division=0)
        if current_f1 > best_subnarr_f1:
            best_subnarr_f1 = current_f1
            best_subnarr_threshold = threshold

    print(f"Best Sub-narrative Threshold: {best_subnarr_threshold:.2f} (F1 Micro: {best_subnarr_f1:.4f})")
    
    # --- Now, let's see the overall F1 score using these two different thresholds ---
    print("\n--- Calculating overall F1 score with per-level thresholds ---")
    
    # Create the final binary prediction matrix
    final_binary_preds = np.zeros_like(probabilities, dtype=int)
    
    # Apply the best thresholds to their respective columns
    final_binary_preds[:, narrative_indices] = (narr_probs > best_narr_threshold).astype(int)
    final_binary_preds[:, subnarrative_indices] = (subnarr_probs > best_subnarr_threshold).astype(int)
    
    # Apply final hierarchical correction
    if parent_child_pairs:
        for sub_id, narr_id in parent_child_pairs:
            inconsistent_mask = (final_binary_preds[:, sub_id] == 1) & (final_binary_preds[:, narr_id] == 0)
            final_binary_preds[inconsistent_mask, sub_id] = 0
            
    overall_f1_micro = f1_score(true_labels, final_binary_preds, average='micro', zero_division=0)
    print(f"Overall F1 Micro with combined per-level thresholds: {overall_f1_micro:.4f}")

    return {
        "narrative_threshold": best_narr_threshold,
        "subnarrative_threshold": best_subnarr_threshold,
        "overall_f1_micro": overall_f1_micro
    }