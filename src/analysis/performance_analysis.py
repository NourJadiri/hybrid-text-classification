#!/usr/bin/env python3
"""
Performance Analysis Script for Hierarchical Text Classification
Computes F1 scores (macro and micro) and confusion matrices for multi-label classification.
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
    """Compute F1 macro and micro scores."""
    # Transform to binary format
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # Compute F1 scores
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1_samples = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    
    return f1_macro, f1_micro, f1_samples

def create_confusion_matrices(y_true, y_pred, mlb, label_type="narratives"):
    """Create confusion matrices for each label."""
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    confusion_matrices = {}
    for i, label in enumerate(mlb.classes_):
        cm = confusion_matrix(y_true_bin[:, i], y_pred_bin[:, i])
        confusion_matrices[label] = cm
    
    return confusion_matrices

def analyze_language(devset_dir, predictions_dir, language, threshold_folder):
    """Analyze performance for a single language."""
    print(f"\n{'='*60}")
    print(f"Analyzing language: {language}")
    print(f"{'='*60}")
    
    # Load annotations
    annotation_file = os.path.join(devset_dir, language, 'subtask-2-annotations.txt')
    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file not found for {language}: {annotation_file}")
        return None
    
    annotations = load_annotations(annotation_file)
    print(f"Loaded {len(annotations)} ground truth annotations")
    
    # Load predictions
    prediction_file = os.path.join(predictions_dir, threshold_folder, f"{language}_predictions.tsv")
    if not os.path.exists(prediction_file):
        print(f"Warning: Prediction file not found for {language}: {prediction_file}")
        return None
    
    predictions = load_predictions(prediction_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Find common files
    common_files = set(annotations.keys()) & set(predictions.keys())
    if not common_files:
        print(f"Warning: No common files found between annotations and predictions for {language}")
        return None
    
    print(f"Found {len(common_files)} files with both annotations and predictions")
    
    # Prepare data for metrics computation
    results = {
        'language': language,
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
        
        # Compute metrics
        f1_macro, f1_micro, f1_samples = compute_metrics(y_true, y_pred, mlb, label_type)
        
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Samples: {f1_samples:.4f}")
        
        # Store results
        results[label_type] = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_samples': f1_samples,
            'total_labels': len(mlb.classes_),
            'label_distribution': dict(Counter([label for labels in y_true for label in labels]))
        }
        
        # Create confusion matrices for most frequent labels (top 10)
        label_counts = Counter([label for labels in y_true for label in labels])
        top_labels = [label for label, _ in label_counts.most_common(10)]
        
        confusion_matrices = create_confusion_matrices(y_true, y_pred, mlb, label_type)
        
        # Save confusion matrices for top labels
        results[label_type]['confusion_matrices'] = {}
        for label in top_labels:
            if label in confusion_matrices:
                cm = confusion_matrices[label]
                results[label_type]['confusion_matrices'][label] = {
                    'matrix': cm.tolist(),
                    'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                    'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                    'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                    'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0
                }
        
        # Print top 5 most frequent labels
        print(f"Top 5 most frequent {label_type}:")
        for i, (label, count) in enumerate(label_counts.most_common(5), 1):
            print(f"  {i}. {label}: {count}")
    
    return results

def plot_confusion_matrix_summary(all_results, output_dir, threshold_folder):
    """Create summary plots of confusion matrices."""
    os.makedirs(os.path.join(output_dir, threshold_folder, 'plots'), exist_ok=True)
    
    for label_type in ['narratives', 'subnarratives']:
        # Collect all confusion matrix data
        all_cms = {}
        
        for lang_result in all_results:
            if lang_result and label_type in lang_result:
                lang = lang_result['language']
                cms = lang_result[label_type].get('confusion_matrices', {})
                for label, cm_data in cms.items():
                    if label not in all_cms:
                        all_cms[label] = {'languages': [], 'precision': [], 'recall': [], 'f1': []}
                    
                    # Calculate precision, recall, F1 for this label
                    tp = cm_data['tp']
                    fp = cm_data['fp']
                    fn = cm_data['fn']
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    all_cms[label]['languages'].append(lang)
                    all_cms[label]['precision'].append(precision)
                    all_cms[label]['recall'].append(recall)
                    all_cms[label]['f1'].append(f1)
        
        # Create individual plots instead of 2x2 subplots
        if all_cms:
            plots_dir = os.path.join(output_dir, threshold_folder, 'plots', label_type)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: F1 scores by label
            plt.figure(figsize=(10, 8))
            labels = list(all_cms.keys())[:10]  # Top 10 labels
            avg_f1s = [np.mean(all_cms[label]['f1']) for label in labels]
            
            plt.barh(labels, avg_f1s)
            plt.xlabel('Average F1 Score')
            plt.title(f'{label_type.capitalize()} - Average F1 Score by Label', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, 'f1_score_by_label.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_file}")
            
            # Plot 2: Performance by language
            plt.figure(figsize=(10, 6))
            languages = set()
            for cms in all_cms.values():
                languages.update(cms['languages'])
            languages = sorted(list(languages))
            
            lang_avg_f1 = []
            for lang in languages:
                f1_scores = []
                for cms in all_cms.values():
                    for i, l in enumerate(cms['languages']):
                        if l == lang:
                            f1_scores.append(cms['f1'][i])
                lang_avg_f1.append(np.mean(f1_scores) if f1_scores else 0)
            
            plt.bar(languages, lang_avg_f1)
            plt.ylabel('Average F1 Score')
            plt.title(f'{label_type.capitalize()} - Average F1 Score by Language', fontsize=14, fontweight='bold')
            plt.ylim(0, 1)
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, 'f1_score_by_language.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_file}")
            
            # Plot 3: Precision vs Recall scatter
            plt.figure(figsize=(8, 8))
            all_precision = []
            all_recall = []
            for cms in all_cms.values():
                all_precision.extend(cms['precision'])
                all_recall.extend(cms['recall'])
            
            plt.scatter(all_recall, all_precision, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{label_type.capitalize()} - Precision vs Recall', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, 'precision_vs_recall.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_file}")
            
            # Plot 4: F1 distribution
            plt.figure(figsize=(10, 6))
            all_f1 = []
            for cms in all_cms.values():
                all_f1.extend(cms['f1'])
            
            plt.hist(all_f1, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('F1 Score')
            plt.ylabel('Frequency')
            plt.title(f'{label_type.capitalize()} - F1 Score Distribution', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, 'f1_score_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze performance of hierarchical text classification predictions")
    parser.add_argument("--devset_dir", type=str, default="../../devset", 
                       help="Path to devset directory")
    parser.add_argument("--predictions_dir", type=str, default="../../results/hub_inference",
                       help="Path to predictions directory")
    parser.add_argument("--threshold_folder", type=str, default="0.65",
                       help="Threshold folder name containing predictions")
    parser.add_argument("--output_dir", type=str, default="../../results/analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--languages", nargs="+", default=["BG", "EN", "HI", "PT", "RU"],
                       help="Languages to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, args.threshold_folder), exist_ok=True)
    
    print("="*80)
    print("HIERARCHICAL TEXT CLASSIFICATION PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Devset directory: {args.devset_dir}")
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Threshold folder: {args.threshold_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Languages: {', '.join(args.languages)}")
    
    # Analyze each language
    all_results = []
    overall_metrics = {
        'narratives': {'f1_macro': [], 'f1_micro': [], 'f1_samples': []},
        'subnarratives': {'f1_macro': [], 'f1_micro': [], 'f1_samples': []}
    }
    
    for language in args.languages:
        result = analyze_language(args.devset_dir, args.predictions_dir, language, args.threshold_folder)
        if result:
            all_results.append(result)
            for label_type in ['narratives', 'subnarratives']:
                overall_metrics[label_type]['f1_macro'].append(result[label_type]['f1_macro'])
                overall_metrics[label_type]['f1_micro'].append(result[label_type]['f1_micro'])
                overall_metrics[label_type]['f1_samples'].append(result[label_type]['f1_samples'])
    
    # Compute overall statistics
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    summary_results = {}
    for label_type in ['narratives', 'subnarratives']:
        print(f"\n{label_type.capitalize()} - Overall Performance:")
        
        metrics = overall_metrics[label_type]
        macro_mean = np.mean(metrics['f1_macro'])
        macro_std = np.std(metrics['f1_macro'])
        micro_mean = np.mean(metrics['f1_micro'])
        micro_std = np.std(metrics['f1_micro'])
        samples_mean = np.mean(metrics['f1_samples'])
        samples_std = np.std(metrics['f1_samples'])
        
        print(f"  F1 Macro:   {macro_mean:.4f} ± {macro_std:.4f}")
        print(f"  F1 Micro:   {micro_mean:.4f} ± {micro_std:.4f}")
        print(f"  F1 Samples: {samples_mean:.4f} ± {samples_std:.4f}")
        
        summary_results[label_type] = {
            'f1_macro_mean': macro_mean,
            'f1_macro_std': macro_std,
            'f1_micro_mean': micro_mean,
            'f1_micro_std': micro_std,
            'f1_samples_mean': samples_mean,
            'f1_samples_std': samples_std
        }
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, args.threshold_folder, 'detailed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold': args.threshold_folder,
            'overall_summary': summary_results,
            'by_language': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        for label_type in ['narratives', 'subnarratives']:
            summary_data.append({
                'Language': result['language'],
                'Label_Type': label_type,
                'F1_Macro': result[label_type]['f1_macro'],
                'F1_Micro': result[label_type]['f1_micro'],
                'F1_Samples': result[label_type]['f1_samples'],
                'Total_Labels': result[label_type]['total_labels'],
                'Total_Files': result['total_files']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(args.output_dir, args.threshold_folder, 'performance_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    # Create plots
    print("\nGenerating performance plots...")
    plot_confusion_matrix_summary(all_results, args.output_dir, args.threshold_folder)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved in: {os.path.join(args.output_dir, args.threshold_folder)}")

if __name__ == "__main__":
    main()