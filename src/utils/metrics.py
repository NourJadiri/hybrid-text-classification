import numpy as np
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