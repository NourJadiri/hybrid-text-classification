# src/utils/balanced_sampling.py

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings

def analyze_class_distribution(df: pd.DataFrame, label_column: str = 'narratives') -> Dict:
    """
    Analyze the distribution of classes in the dataset.
    
    Args:
        df: DataFrame with the dataset
        label_column: Column containing the labels (list of narratives)
        
    Returns:
        Dictionary with class distribution statistics
    """
    # Flatten all narratives to get individual class counts
    all_narratives = []
    for narratives in df[label_column]:
        if isinstance(narratives, list):
            all_narratives.extend(narratives)
        elif isinstance(narratives, np.ndarray):
            # Handle numpy arrays
            all_narratives.extend(narratives.tolist())
        elif pd.notna(narratives) and not isinstance(narratives, (list, np.ndarray)):
            all_narratives.append(narratives)
    
    class_counts = Counter(all_narratives)
    total_samples = len(df)
    
    # Calculate statistics
    stats = {
        'total_samples': total_samples,
        'class_counts': dict(class_counts),
        'unique_classes': len(class_counts),
        'most_common': class_counts.most_common(5),
        'least_common': class_counts.most_common()[-5:] if len(class_counts) >= 5 else class_counts.most_common()
    }
    
    # Check for "Other" class dominance
    other_variants = ['Other', 'other', 'OTHER', 'None', 'none']
    other_count = sum(class_counts.get(variant, 0) for variant in other_variants)
    stats['other_count'] = other_count
    stats['other_percentage'] = (other_count / sum(class_counts.values())) * 100
    
    return stats


def create_balanced_sample_strategy(
    df: pd.DataFrame,
    label_column: str = 'narratives',
    target_size: int = 1000,
    min_samples_per_class: int = 10,
    max_other_percentage: float = 0.2,
    prioritize_rare_classes: bool = True,
    random_state: int = 42
) -> Dict[str, int]:
    """
    Create a sampling strategy that maximizes class diversity while minimizing "Other" class.
    
    Args:
        df: DataFrame with the dataset
        label_column: Column containing the labels (list of narratives)
        target_size: Total number of samples desired
        min_samples_per_class: Minimum samples per class (for rare classes)
        max_other_percentage: Maximum percentage of "Other" class allowed
        prioritize_rare_classes: Whether to prioritize rare classes over common ones
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary mapping class names to number of samples to include
    """
    # Analyze current distribution
    stats = analyze_class_distribution(df, label_column)
    class_counts = stats['class_counts']
    
    # Identify "Other" variants
    other_variants = ['Other', 'other', 'OTHER', 'None', 'none']
    other_classes = [cls for cls in class_counts.keys() if cls in other_variants]
    non_other_classes = [cls for cls in class_counts.keys() if cls not in other_variants]
    
    # Calculate target distribution
    max_other_samples = int(target_size * max_other_percentage)
    remaining_samples = target_size - max_other_samples
    
    # Create sampling strategy
    sampling_strategy = {}
    
    # Handle non-"Other" classes
    if prioritize_rare_classes:
        # Sort classes by frequency (ascending) to prioritize rare classes
        sorted_classes = sorted(non_other_classes, key=lambda x: class_counts[x])
    else:
        # Sort classes by frequency (descending) to prioritize common classes
        sorted_classes = sorted(non_other_classes, key=lambda x: class_counts[x], reverse=True)
    
    # Distribute samples among non-"Other" classes
    samples_per_class = max(min_samples_per_class, remaining_samples // max(len(non_other_classes), 1))
    
    allocated_samples = 0
    for cls in sorted_classes:
        if allocated_samples >= remaining_samples:
            break
            
        # Ensure we don't exceed available samples for this class
        available_samples = class_counts[cls]
        target_samples = min(samples_per_class, available_samples, remaining_samples - allocated_samples)
        
        if target_samples > 0:
            sampling_strategy[cls] = target_samples
            allocated_samples += target_samples
    
    # Handle "Other" classes with remaining budget
    remaining_other_budget = min(max_other_samples, target_size - allocated_samples)
    if remaining_other_budget > 0 and other_classes:
        # Distribute among "Other" variants
        samples_per_other = remaining_other_budget // len(other_classes)
        remaining_other = remaining_other_budget % len(other_classes)
        
        for i, cls in enumerate(other_classes):
            available_samples = class_counts[cls]
            target_samples = samples_per_other + (1 if i < remaining_other else 0)
            target_samples = min(target_samples, available_samples)
            
            if target_samples > 0:
                sampling_strategy[cls] = target_samples
    
    return sampling_strategy


def create_balanced_multilabel_sample(
    df: pd.DataFrame,
    label_column: str = 'narratives',
    text_column: str = 'text',
    target_size: int = 1000,
    min_samples_per_class: int = 10,
    max_other_percentage: float = 0.2,
    prioritize_rare_classes: bool = True,
    random_state: int = 42,
    stratify_by_primary_class: bool = True
) -> pd.DataFrame:
    """
    Create a balanced sample from a multilabel dataset, minimizing "Other" class
    and maximizing class diversity.
    
    Args:
        df: DataFrame with the dataset
        label_column: Column containing the labels (list of narratives)
        text_column: Column containing the text
        target_size: Total number of samples desired
        min_samples_per_class: Minimum samples per class (for rare classes)
        max_other_percentage: Maximum percentage of "Other" class allowed
        prioritize_rare_classes: Whether to prioritize rare classes over common ones
        random_state: Random state for reproducibility
        stratify_by_primary_class: Whether to stratify by the primary (first) class
        
    Returns:
        Balanced sample DataFrame
    """
    np.random.seed(random_state)
    
    # Clean data
    df_clean = df[[text_column, label_column]].dropna().copy()
    
    print(f"Original dataset size: {len(df_clean)}")
    
    # Analyze distribution
    stats = analyze_class_distribution(df_clean, label_column)
    print(f"Found {stats['unique_classes']} unique classes")
    print(f"'Other' class percentage: {stats['other_percentage']:.1f}%")
    print(f"Most common classes: {stats['most_common']}")
    
    # Create sampling strategy
    sampling_strategy = create_balanced_sample_strategy(
        df_clean, label_column, target_size, min_samples_per_class,
        max_other_percentage, prioritize_rare_classes, random_state
    )
    
    print(f"\nSampling strategy: {sampling_strategy}")
    
    # Create class-specific samples
    sampled_indices = set()
    class_samples = {}
    
    # For each target class, find samples containing that class
    for target_class, target_count in sampling_strategy.items():
        # Find all samples that contain this class
        def contains_class(x):
            if isinstance(x, list):
                return target_class in x
            elif isinstance(x, np.ndarray):
                return target_class in x.tolist()
            else:
                return False
                
        class_mask = df_clean[label_column].apply(contains_class)
        class_samples_df = df_clean[class_mask]
        
        print(f"\nClass '{target_class}': {len(class_samples_df)} available, targeting {target_count}")
        
        if len(class_samples_df) == 0:
            continue
            
        # Sample from available samples
        n_samples = min(target_count, len(class_samples_df))
        if n_samples > 0:
            sampled_class_df = class_samples_df.sample(n=n_samples, random_state=random_state)
            class_samples[target_class] = sampled_class_df
            sampled_indices.update(sampled_class_df.index)
    
    # If we haven't reached target size, add more diverse samples
    if len(sampled_indices) < target_size:
        remaining_needed = target_size - len(sampled_indices)
        
        # Get samples not yet included
        remaining_df = df_clean[~df_clean.index.isin(sampled_indices)]
        
        if len(remaining_df) > 0:
            # Prioritize samples with multiple rare classes
            def diversity_score(narratives):
                if isinstance(narratives, list):
                    narrative_list = narratives
                elif isinstance(narratives, np.ndarray):
                    narrative_list = narratives.tolist()
                else:
                    return 0
                
                score = 0
                for narrative in narrative_list:
                    if narrative not in ['Other', 'other', 'OTHER', 'None', 'none']:
                        # Give higher score to rarer classes
                        class_freq = stats['class_counts'].get(narrative, 1)
                        score += 1.0 / (class_freq + 1)  # +1 to avoid division by zero
                
                return score
            
            remaining_df = remaining_df.copy()
            remaining_df['diversity_score'] = remaining_df[label_column].apply(diversity_score)
            remaining_df = remaining_df.sort_values('diversity_score', ascending=False)
            
            # Sample additional diverse samples
            additional_samples = min(remaining_needed, len(remaining_df))
            additional_df = remaining_df.head(additional_samples)
            sampled_indices.update(additional_df.index)
    
    # Create final balanced sample
    balanced_sample = df_clean.loc[list(sampled_indices)].copy()
    
    # Shuffle the final sample
    balanced_sample = balanced_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Print final statistics
    final_stats = analyze_class_distribution(balanced_sample, label_column)
    print(f"\n=== FINAL BALANCED SAMPLE ===")
    print(f"Sample size: {len(balanced_sample)}")
    print(f"Unique classes: {final_stats['unique_classes']}")
    print(f"'Other' class percentage: {final_stats['other_percentage']:.1f}%")
    print(f"Most common classes in sample: {final_stats['most_common']}")
    
    return balanced_sample


def create_stratified_multilabel_split(
    df: pd.DataFrame,
    label_column: str = 'narratives',
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split for multilabel data.
    
    Args:
        df: DataFrame with the dataset
        label_column: Column containing the labels (list of narratives)
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # For multilabel data, we'll stratify by the primary class (first in the list)
    df = df.copy()
    def get_primary_class(x):
        if isinstance(x, list) and len(x) > 0:
            return x[0]
        elif isinstance(x, np.ndarray) and len(x) > 0:
            return x[0]
        else:
            return 'Other'
            
    df['primary_class'] = df[label_column].apply(get_primary_class)
    
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['primary_class']
        )
        
        # Remove the temporary column
        train_df = train_df.drop('primary_class', axis=1)
        test_df = test_df.drop('primary_class', axis=1)
        
    except ValueError as e:
        # Fall back to random split if stratification fails
        warnings.warn(f"Stratification failed ({e}), using random split instead")
        train_df, test_df = train_test_split(
            df.drop('primary_class', axis=1), 
            test_size=test_size, 
            random_state=random_state
        )
    
    return train_df, test_df


# Example usage function
def create_balanced_sample_for_finetuning(
    dataset_path: str,
    target_size: int = 1000,
    max_other_percentage: float = 0.15,
    output_path: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a balanced sample specifically for finetuning, with minimal "Other" class.
    
    Args:
        dataset_path: Path to the parquet dataset file
        target_size: Target number of samples
        max_other_percentage: Maximum percentage of "Other" class (default 15%)
        output_path: Optional path to save the balanced sample
        random_state: Random state for reproducibility
        
    Returns:
        Balanced sample DataFrame
    """
    # Import here to avoid circular imports
    from data_management.loaders import load_labeled_df
    
    print("Loading dataset...")
    df = load_labeled_df(dataset_path)
    
    print(f"Original dataset: {len(df)} samples")
    
    # Create balanced sample
    balanced_sample = create_balanced_multilabel_sample(
        df=df,
        label_column='narratives',
        text_column='text',
        target_size=target_size,
        min_samples_per_class=5,  # Minimum 5 samples per class
        max_other_percentage=max_other_percentage,
        prioritize_rare_classes=True,  # Focus on rare, interesting classes
        random_state=random_state
    )
    
    # Save if output path provided
    if output_path:
        balanced_sample.to_parquet(output_path, index=False)
        print(f"\nBalanced sample saved to: {output_path}")
    
    return balanced_sample


if __name__ == "__main__":
    # Example usage
    sample_df = create_balanced_sample_for_finetuning(
        dataset_path="data/processed/phase0_baseline_labeled.parquet",
        target_size=1000,
        max_other_percentage=0.15,
        output_path="data/processed/balanced_sample_for_finetuning.parquet"
    )
    
    print(f"\nCreated balanced sample with {len(sample_df)} samples")
