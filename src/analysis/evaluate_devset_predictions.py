#!/usr/bin/env python3
"""
Generic Performance Analysis Script for Hierarchical Text Classification
Evaluates predictions against ground truth for devset with comprehensive metrics and visualizations.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json

def parse_labels(label_string):
    """Parse semicolon-separated labels into a list."""
    if label_string.strip() == "Other":
        return ["Other"]
    return [label.strip() for label in label_string.split(";") if label.strip()]

def load_annotations(annotation_file):
    """Load ground truth annotations from file."""
    annotations = {}
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                filename = parts[0]
                narratives = parse_labels(parts[1])
                subnarratives = parse_labels(parts[2])
                annotations[filename] = {
                    'narratives': narratives,
                    'subnarratives': subnarratives
                }
    return annotations

def load_predictions(prediction_file):
    """Load predictions from file."""
    predictions = {}
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                filename = parts[0]
                narratives = parse_labels(parts[1])
                subnarratives = parse_labels(parts[2])
                predictions[filename] = {
                    'narratives': narratives,
                    'subnarratives': subnarratives
                }
    return predictions

def compute_metrics(y_true, y_pred, mlb, label_type="narratives"):
    """Compute F1 macro, micro, and samples scores."""
    # Transform to binary format
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # Compute F1 scores
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1_samples = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    
    return f1_macro, f1_micro, f1_samples

def compute_per_label_metrics(y_true, y_pred, mlb):
    """Compute per-label precision, recall, and F1 scores."""
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    per_label_metrics = {}
    for i, label in enumerate(mlb.classes_):
        y_true_label = y_true_bin[:, i]
        y_pred_label = y_pred_bin[:, i]
        
        tp = np.sum((y_true_label == 1) & (y_pred_label == 1))
        fp = np.sum((y_true_label == 0) & (y_pred_label == 1))
        fn = np.sum((y_true_label == 1) & (y_pred_label == 0))
        tn = np.sum((y_true_label == 0) & (y_pred_label == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_label_metrics[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(y_true_label == 1)),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    return per_label_metrics

def create_confusion_matrices(y_true, y_pred, mlb, label_type="narratives"):
    """Create confusion matrices for each label."""
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    confusion_matrices = {}
    for i, label in enumerate(mlb.classes_):
        cm = confusion_matrix(y_true_bin[:, i], y_pred_bin[:, i])
        confusion_matrices[label] = cm
    
    return confusion_matrices

def analyze_single_evaluation(ground_truth_file, prediction_file, language_code, model_name="Unknown"):
    """Analyze performance for a single evaluation."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {language_code} ({model_name})")
    print(f"{'='*60}")
    print(f"Ground truth: {ground_truth_file}")
    print(f"Predictions: {prediction_file}")
    
    # Load annotations
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return None
    
    annotations = load_annotations(ground_truth_file)
    print(f"Loaded {len(annotations)} ground truth annotations")
    
    # Load predictions
    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file not found: {prediction_file}")
        return None
    
    predictions = load_predictions(prediction_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Find common files
    common_files = set(annotations.keys()) & set(predictions.keys())
    if not common_files:
        print(f"Error: No common files found between annotations and predictions")
        return None
    
    print(f"Found {len(common_files)} files with both annotations and predictions")
    
    # Prepare data for metrics computation
    results = {
        'language': language_code,
        'model': model_name,
        'total_files': len(common_files),
        'narratives': {},
        'subnarratives': {}
    }
    
    for label_type in ['narratives', 'subnarratives']:
        print(f"\n--- {label_type.capitalize()} Analysis ---")
        
        # Extract labels
        y_true = [annotations[f][label_type] for f in common_files]
        y_pred = [predictions[f][label_type] for f in common_files]
        
        # Create MultiLabelBinarizer
        all_labels = set()
        for labels in y_true + y_pred:
            all_labels.update(labels)
        
        mlb = MultiLabelBinarizer()
        mlb.fit([list(all_labels)])
        
        print(f"Total unique {label_type}: {len(mlb.classes_)}")
        
        # Compute overall metrics
        f1_macro, f1_micro, f1_samples = compute_metrics(y_true, y_pred, mlb, label_type)
        
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Samples: {f1_samples:.4f}")
        
        # Compute per-label metrics
        per_label_metrics = compute_per_label_metrics(y_true, y_pred, mlb)
        
        # Store results
        results[label_type] = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_samples': f1_samples,
            'total_labels': len(mlb.classes_),
            'label_distribution': dict(Counter([label for labels in y_true for label in labels])),
            'per_label_metrics': per_label_metrics,
            'all_labels': list(mlb.classes_)
        }
        
        # Create confusion matrices for all labels
        confusion_matrices = create_confusion_matrices(y_true, y_pred, mlb, label_type)
        
        # Save confusion matrices
        results[label_type]['confusion_matrices'] = {}
        for label in mlb.classes_:
            if label in confusion_matrices:
                cm = confusion_matrices[label]
                results[label_type]['confusion_matrices'][label] = {
                    'matrix': cm.tolist(),
                    'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                    'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                    'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                    'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0
                }
        
        # Print label ranking by F1 score
        label_f1_ranking = sorted(per_label_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        print(f"\nTop 10 {label_type} by F1 score:")
        for i, (label, metrics) in enumerate(label_f1_ranking[:10], 1):
            print(f"  {i:2d}. {label:<50} F1: {metrics['f1']:.4f} (P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, Support: {metrics['support']})")
        
        # Print label ranking by support (frequency)
        label_support_ranking = sorted(per_label_metrics.items(), key=lambda x: x[1]['support'], reverse=True)
        print(f"\nTop 10 most frequent {label_type}:")
        for i, (label, metrics) in enumerate(label_support_ranking[:10], 1):
            print(f"  {i:2d}. {label:<50} Support: {metrics['support']:<3d} F1: {metrics['f1']:.4f}")
    
    return results

def create_performance_plots(all_results, output_dir, experiment_name="evaluation"):
    """Create comprehensive performance plots."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for label_type in ['narratives', 'subnarratives']:
        # Language comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{label_type.capitalize()} - Performance Analysis ({experiment_name})', fontsize=16, fontweight='bold')
        
        languages = [r['language'] for r in all_results if r]
        f1_macro_scores = [r[label_type]['f1_macro'] for r in all_results if r]
        f1_micro_scores = [r[label_type]['f1_micro'] for r in all_results if r]
        f1_samples_scores = [r[label_type]['f1_samples'] for r in all_results if r]
        
        # F1 Macro comparison
        bars1 = axes[0, 0].bar(languages, f1_macro_scores, color=colors[:len(languages)])
        axes[0, 0].set_title('F1 Macro Score by Language')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim(0, 1)
        for bar, v in zip(bars1, f1_macro_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 Micro comparison
        bars2 = axes[0, 1].bar(languages, f1_micro_scores, color=colors[:len(languages)])
        axes[0, 1].set_title('F1 Micro Score by Language')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_ylim(0, 1)
        for bar, v in zip(bars2, f1_micro_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 Samples comparison
        bars3 = axes[1, 0].bar(languages, f1_samples_scores, color=colors[:len(languages)])
        axes[1, 0].set_title('F1 Samples Score by Language')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_ylim(0, 1)
        for bar, v in zip(bars3, f1_samples_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Combined comparison
        x = np.arange(len(languages))
        width = 0.25
        axes[1, 1].bar(x - width, f1_macro_scores, width, label='F1 Macro', color='#1f77b4')
        axes[1, 1].bar(x, f1_micro_scores, width, label='F1 Micro', color='#ff7f0e')
        axes[1, 1].bar(x + width, f1_samples_scores, width, label='F1 Samples', color='#2ca02c')
        axes[1, 1].set_title('F1 Scores Comparison')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(languages)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'{label_type}_language_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_file}")
        
        # Top labels performance across all languages
        all_labels_metrics = {}
        for result in all_results:
            if result and label_type in result:
                for label, metrics in result[label_type]['per_label_metrics'].items():
                    if label not in all_labels_metrics:
                        all_labels_metrics[label] = {'f1_scores': [], 'supports': []}
                    all_labels_metrics[label]['f1_scores'].append(metrics['f1'])
                    all_labels_metrics[label]['supports'].append(metrics['support'])
        
        # Calculate average F1 and total support for each label
        label_avg_metrics = {}
        for label, data in all_labels_metrics.items():
            label_avg_metrics[label] = {
                'avg_f1': np.mean(data['f1_scores']),
                'total_support': sum(data['supports'])
            }
        
        # Top labels by average F1 score
        top_labels_f1 = sorted(label_avg_metrics.items(), 
                              key=lambda x: x[1]['avg_f1'], reverse=True)[:15]
        
        plt.figure(figsize=(12, 8))
        labels = [item[0][:50] for item in top_labels_f1]  # Truncate long labels
        f1_scores = [item[1]['avg_f1'] for item in top_labels_f1]
        
        plt.barh(range(len(labels)), f1_scores)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Average F1 Score')
        plt.title(f'{label_type.capitalize()} - Top 15 Labels by Average F1 Score ({experiment_name})', 
                 fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, f'{label_type}_top_labels_f1.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical text classification predictions against ground truth")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth annotations file")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions file")
    parser.add_argument("--language", type=str, required=True,
                       help="Language code (e.g., EN, BG, HI, PT, RU)")
    parser.add_argument("--model_name", type=str, default="Unknown Model",
                       help="Name of the model being evaluated")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--experiment_name", type=str, default="evaluation",
                       help="Name of this evaluation experiment")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("HIERARCHICAL TEXT CLASSIFICATION - DEVSET EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Language: {args.language}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Predictions: {args.predictions}")
    print(f"Output directory: {args.output_dir}")
    
    # Analyze single evaluation
    result = analyze_single_evaluation(
        args.ground_truth, 
        args.predictions, 
        args.language, 
        args.model_name
    )
    
    if not result:
        print("Evaluation failed!")
        return
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f'{args.language}_detailed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': args.experiment_name,
            'model': args.model_name,
            'language': args.language,
            'ground_truth_file': args.ground_truth,
            'predictions_file': args.predictions,
            'results': result
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create summary CSV
    summary_data = []
    for label_type in ['narratives', 'subnarratives']:
        summary_data.append({
            'Language': result['language'],
            'Model': result['model'],
            'Label_Type': label_type,
            'F1_Macro': result[label_type]['f1_macro'],
            'F1_Micro': result[label_type]['f1_micro'],
            'F1_Samples': result[label_type]['f1_samples'],
            'Total_Labels': result[label_type]['total_labels'],
            'Total_Files': result['total_files']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(args.output_dir, f'{args.language}_performance_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    # Create plots for single language
    print("\nGenerating performance plots...")
    create_performance_plots([result], args.output_dir, args.experiment_name)
    
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()