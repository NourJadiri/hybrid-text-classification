from typing import Counter
import numpy as np
import pandas as pd
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
    
def get_class_distribution(dataframe: pd.DataFrame, id_to_label_map: dict) -> pd.DataFrame:
    """
    Calculates the frequency of each label in a DataFrame.
    (This function remains the same as before, as it only depends on the DataFrame)
    """
    print("Calculating class distribution...")
    # ... (implementation is identical to the previous version) ...
    if 'narrative_ids' not in dataframe.columns or 'subnarrative_ids' not in dataframe.columns:
        raise ValueError("Input DataFrame must contain 'narrative_ids' and 'subnarrative_ids' columns.")
    all_label_ids = [id_val for sublist in dataframe['narrative_ids'].dropna() if isinstance(sublist, list) for id_val in sublist] + \
                    [id_val for sublist in dataframe['subnarrative_ids'].dropna() if isinstance(sublist, list) for id_val in sublist]
    if not all_label_ids:
        print("Warning: No labels found in the provided DataFrame.")
        return pd.DataFrame(columns=['label', 'count', 'level'])
    label_id_counts = Counter(all_label_ids)
    dist_data = []
    for id_val, count in label_id_counts.items():
        label_str = id_to_label_map.get(id_val, f"Unknown ID: {id_val}")
        colon_count = label_str.count(':')
        level = 'Narrative' if colon_count == 1 else 'Sub-narrative' if colon_count == 2 else 'Other'
        dist_data.append({'label': label_str, 'count': count, 'level': level})
    class_counts_df = pd.DataFrame(dist_data).sort_values(by='count', ascending=True)
    print("Class distribution calculation complete.")
    return class_counts_df


# --- THIS IS THE CORRECTED FUNCTION ---
def get_per_class_f1_scores(
    true_labels: np.ndarray, 
    pred_logits: np.ndarray, 
    id_to_label_map: dict, 
    narrative_indices: list,
    subnarrative_indices: list,
    narrative_threshold: float,
    subnarrative_threshold: float
) -> pd.DataFrame:
    """
    Calculates the F1 score for each individual class from raw logits,
    using separate thresholds for narratives and sub-narratives.

    Args:
        true_labels (np.ndarray): The ground truth multi-hot encoded labels.
        pred_logits (np.ndarray): The raw logit predictions from the model.
        id_to_label_map (dict): The dictionary mapping integer IDs to label strings.
        narrative_indices (list): List of column indices for narrative labels.
        subnarrative_indices (list): List of column indices for sub-narrative labels.
        narrative_threshold (float): The optimal threshold for narrative labels.
        subnarrative_threshold (float): The optimal threshold for sub-narrative labels.

    Returns:
        pd.DataFrame: A DataFrame with 'label', 'f1_score', sorted by F1 score.
    """
    print("Calculating per-class F1 scores with per-level thresholds...")
    
    # --- THIS IS THE KEY CHANGE ---
    # Convert logits to probabilities
    pred_probs = 1 / (1 + np.exp(-pred_logits))
    
    print(f"Probabilities Stats: Min={np.min(pred_probs):.4f}, Max={np.max(pred_probs):.4f}, Mean={np.mean(pred_probs):.4f}")
    print(f"Narrative Threshold: {narrative_threshold}, Sub-narrative Threshold: {subnarrative_threshold}")

    # Initialize the binary predictions matrix with zeros
    binary_preds = np.zeros_like(pred_probs, dtype=int)

    # Apply the separate thresholds to their respective columns
    binary_preds[:, narrative_indices] = (pred_probs[:, narrative_indices] > narrative_threshold).astype(int)
    binary_preds[:, subnarrative_indices] = (pred_probs[:, subnarrative_indices] > subnarrative_threshold).astype(int)
    # -----------------------------
    
    
    num_positive_preds = np.sum(binary_preds)
    print(f"Total number of positive predictions made (after thresholding): {num_positive_preds}")
    if num_positive_preds == 0:
        print("!! WARNING: No positive predictions were made. This will result in an F1 score of 0.")
    
    # Calculate F1 score for each class (average=None returns an array of scores)
    per_class_f1 = f1_score(true_labels, binary_preds, average=None, zero_division=0)
    
    
    # Create a DataFrame for easy viewing
    f1_scores_df = pd.DataFrame([
        {'label': id_to_label_map.get(i, f"Unknown ID: {i}"), 'f1_score': score}
        for i, score in enumerate(per_class_f1)
    ])
    
    f1_scores_df = f1_scores_df.sort_values(by='f1_score', ascending=True)
    
    print("Per-class F1 score calculation complete.")
    return f1_scores_df