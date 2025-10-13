#!/usr/bin/env python3
"""
Simple Multi-label Confusion Heatmap for Narratives
Shows which narrative labels are confused with which
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_annotation_line(line):
    """Parse a single annotation line to extract filename and narrative labels"""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, []
    
    filename = parts[0]
    
    # Extract narrative labels (e.g., "CC: Criticism of climate movement")
    if parts[1] != 'Other':
        narratives = [label.strip() for label in parts[1].split(';')]
        labels = list(set(narratives))  # Remove duplicates
    else:
        labels = ['Other']
    
    return filename, labels

def load_annotations(filepath):
    """Load annotations from file"""
    annotations = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            filename, labels = parse_annotation_line(line)
            if filename:
                annotations[filename] = set(labels)
    
    return annotations

def create_confusion_matrix(ground_truth, predictions):
    """
    Create confusion matrix for multi-label classification
    confusion[i][j] = count where true label i appears and predicted label j appears
    """
    # Get all unique labels
    all_labels = sorted(set.union(
        set([label for labels in ground_truth.values() for label in labels]),
        set([label for labels in predictions.values() for label in labels])
    ))
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    n_labels = len(all_labels)
    
    # Initialize confusion matrix
    confusion = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for filename in ground_truth:
        if filename not in predictions:
            continue
        
        true_labels = ground_truth[filename]
        pred_labels = predictions[filename]
        
        # For each true label, count which predicted labels co-occur
        for true_label in true_labels:
            true_idx = label_to_idx[true_label]
            for pred_label in pred_labels:
                pred_idx = label_to_idx[pred_label]
                confusion[true_idx][pred_idx] += 1
    
    return confusion, all_labels

def plot_confusion_heatmap(confusion_matrix, labels, output_path):
    """Plot confusion heatmap - both raw counts and normalized"""
    
    # Normalize by row (true labels) to show percentages
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_confusion = (confusion_matrix / row_sums * 100).astype(float)
    
    # Determine figure size based on number of labels
    n_labels = len(labels)
    fig_width = max(16, n_labels * 0.8)
    fig_height = max(12, n_labels * 0.6)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width * 2, fig_height))
    
    # Plot 1: Raw counts
    sns.heatmap(confusion_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='d', 
                cmap='YlOrRd',
                ax=ax1,
                square=False,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Narrative Confusion Matrix (Raw Counts)\nRows: True Labels, Columns: Predicted Labels', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Labels', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)
    
    # Plot 2: Normalized percentages
    sns.heatmap(normalized_confusion, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',
                ax=ax2,
                square=False,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Narrative Confusion Matrix (Normalized %)\nShows % of times predicted label appears when true label is present', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Labels', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion heatmap to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate narrative confusion heatmap")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Ground truth annotations file")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Predictions file")
    parser.add_argument("--output_dir", type=str, default="confusion_analysis",
                       help="Output directory for heatmap")
    parser.add_argument("--filter_prefix", type=str, default=None,
                       help="Filter to only show labels with this prefix (e.g., 'CC', 'URW')")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading annotations...")
    ground_truth = load_annotations(args.ground_truth)
    predictions = load_annotations(args.predictions)
    
    print(f"✓ Loaded {len(ground_truth)} ground truth samples")
    print(f"✓ Loaded {len(predictions)} predictions")
    
    # Find common files
    common_files = set(ground_truth.keys()) & set(predictions.keys())
    print(f"✓ Found {len(common_files)} common files for analysis")
    
    if len(common_files) == 0:
        print("ERROR: No common files found between ground truth and predictions!")
        return
    
    print("\nCreating confusion matrix...")
    confusion_matrix, labels = create_confusion_matrix(ground_truth, predictions)
    
    # Filter labels if requested
    if args.filter_prefix:
        print(f"\nFiltering labels with prefix '{args.filter_prefix}' and 'Other'...")
        
        # Find indices of labels to keep
        keep_indices = []
        filtered_labels = []
        for i, label in enumerate(labels):
            if label.startswith(args.filter_prefix) or label == 'Other':
                keep_indices.append(i)
                filtered_labels.append(label)
        
        # Filter confusion matrix
        confusion_matrix = confusion_matrix[np.ix_(keep_indices, keep_indices)]
        labels = filtered_labels
        
        print(f"✓ Filtered to {len(labels)} labels")
    
    print(f"✓ Found {len(labels)} unique narrative labels")
    print(f"\nNarrative labels analyzed:")
    for i, label in enumerate(labels, 1):
        print(f"  {i}. {label}")
    
    print("\nGenerating confusion heatmap...")
    filename_suffix = f"_{args.filter_prefix.lower()}" if args.filter_prefix else ""
    heatmap_path = os.path.join(args.output_dir, f"narrative_confusion_heatmap{filename_suffix}.png")
    plot_confusion_heatmap(confusion_matrix, labels, heatmap_path)
    
    # Print diagonal (correct predictions)
    print("\n" + "=" * 70)
    print("Correct Predictions (Diagonal):")
    print("=" * 70)
    for i, label in enumerate(labels):
        correct = confusion_matrix[i][i]
        total = confusion_matrix[i].sum()
        if total > 0:
            percentage = (correct / total) * 100
            print(f"{label:50s}: {correct:3d}/{total:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Output: {heatmap_path}")

if __name__ == "__main__":
    main()
