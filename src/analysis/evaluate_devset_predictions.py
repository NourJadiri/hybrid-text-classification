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

def analyze_detailed_results_per_language(results_dir, top_n=10):
    """
    Analyze detailed results files for per-language top narratives/subnarratives performance.
    
    Args:
        results_dir: Directory containing the detailed results JSON files
        top_n: Number of top performing labels to show per language
    """
    print(f"\n{'='*80}")
    print("PER-LANGUAGE TOP NARRATIVES/SUBNARRATIVES ANALYSIS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Showing top {top_n} labels per language\n")
    
    # Find all detailed results files
    language_files = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('_detailed_results.json'):
            language_code = filename.split('_')[0]
            language_files[language_code] = os.path.join(results_dir, filename)
    
    if not language_files:
        print("No detailed results files found!")
        return
    
    print(f"Found results for languages: {', '.join(sorted(language_files.keys()))}")
    
    # Process each language
    all_language_results = {}
    
    for language_code in sorted(language_files.keys()):
        filepath = language_files[language_code]
        print(f"\n{'-'*60}")
        print(f"LANGUAGE: {language_code}")
        print(f"{'-'*60}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data['results']
            model_name = data.get('model', 'Unknown')
            total_files = results.get('total_files', 0)
            
            print(f"Model: {model_name}")
            print(f"Total files analyzed: {total_files}")
            
            all_language_results[language_code] = {
                'model': model_name,
                'total_files': total_files,
                'narratives': {},
                'subnarratives': {}
            }
            
            # Analyze narratives and subnarratives
            for label_type in ['narratives', 'subnarratives']:
                if label_type not in results:
                    continue
                    
                print(f"\n--- {label_type.upper()} ---")
                
                label_data = results[label_type]
                f1_macro = label_data.get('f1_macro', 0)
                f1_micro = label_data.get('f1_micro', 0)
                f1_samples = label_data.get('f1_samples', 0)
                total_labels = label_data.get('total_labels', 0)
                
                print(f"Overall F1 Macro: {f1_macro:.4f}")
                print(f"Overall F1 Micro: {f1_micro:.4f}")
                print(f"Overall F1 Samples: {f1_samples:.4f}")
                print(f"Total unique labels: {total_labels}")
                
                # Store overall metrics
                all_language_results[language_code][label_type] = {
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro,
                    'f1_samples': f1_samples,
                    'total_labels': total_labels,
                    'top_labels': []
                }
                
                # Get per-label metrics
                per_label_metrics = label_data.get('per_label_metrics', {})
                
                if per_label_metrics:
                    # Sort by F1 score (descending)
                    sorted_labels = sorted(
                        per_label_metrics.items(),
                        key=lambda x: x[1]['f1'],
                        reverse=True
                    )
                    
                    print(f"\nTop {min(top_n, len(sorted_labels))} {label_type} by F1 Score:")
                    print("Rank | Label | F1 | Precision | Recall | Support | TP | FP | FN | TN")
                    print("-" * 100)
                    
                    for i, (label, metrics) in enumerate(sorted_labels[:top_n], 1):
                        f1 = metrics.get('f1', 0)
                        precision = metrics.get('precision', 0)
                        recall = metrics.get('recall', 0)
                        support = metrics.get('support', 0)
                        tp = metrics.get('tp', 0)
                        fp = metrics.get('fp', 0)
                        fn = metrics.get('fn', 0)
                        tn = metrics.get('tn', 0)
                        
                        # Truncate long labels for display
                        display_label = label[:45] + "..." if len(label) > 48 else label
                        
                        print(f"{i:4d} | {display_label:<48} | {f1:.3f} | {precision:9.3f} | {recall:6.3f} | {support:7d} | {tp:2d} | {fp:2d} | {fn:2d} | {tn:2d}")
                        
                        # Store for summary
                        all_language_results[language_code][label_type]['top_labels'].append({
                            'rank': i,
                            'label': label,
                            'f1': f1,
                            'precision': precision,
                            'recall': recall,
                            'support': support,
                            'tp': tp,
                            'fp': fp,
                            'fn': fn,
                            'tn': tn
                        })
                    
                    # Show labels with highest support (most frequent)
                    sorted_by_support = sorted(
                        per_label_metrics.items(),
                        key=lambda x: x[1]['support'],
                        reverse=True
                    )
                    
                    print(f"\nTop {min(top_n, len(sorted_by_support))} most frequent {label_type}:")
                    print("Rank | Label | Support | F1 | Precision | Recall")
                    print("-" * 80)
                    
                    for i, (label, metrics) in enumerate(sorted_by_support[:top_n], 1):
                        support = metrics.get('support', 0)
                        f1 = metrics.get('f1', 0)
                        precision = metrics.get('precision', 0)
                        recall = metrics.get('recall', 0)
                        
                        # Truncate long labels for display
                        display_label = label[:45] + "..." if len(label) > 48 else label
                        print(f"{i:4d} | {display_label:<48} | {support:7d} | {f1:.3f} | {precision:9.3f} | {recall:6.3f}")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Generate cross-language comparison
    print(f"\n{'='*80}")
    print("CROSS-LANGUAGE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Create summary tables
    for label_type in ['narratives', 'subnarratives']:
        print(f"\n--- {label_type.upper()} COMPARISON ---")
        
        # Overall metrics comparison
        print(f"\nOverall Performance Comparison:")
        print("Language | F1 Macro | F1 Micro | F1 Samples | Total Labels | Files")
        print("-" * 70)
        
        for lang in sorted(all_language_results.keys()):
            data = all_language_results[lang][label_type]
            f1_macro = data['f1_macro']
            f1_micro = data['f1_micro']
            f1_samples = data['f1_samples']
            total_labels = data['total_labels']
            total_files = all_language_results[lang]['total_files']
            
            print(f"{lang:8s} | {f1_macro:8.4f} | {f1_micro:8.4f} | {f1_samples:10.4f} | {total_labels:11d} | {total_files:5d}")
        
        # Best performing labels per language
        print(f"\nBest performing {label_type} per language (Top 3):")
        for lang in sorted(all_language_results.keys()):
            top_labels = all_language_results[lang][label_type]['top_labels'][:3]
            print(f"\n{lang}:")
            for i, label_info in enumerate(top_labels, 1):
                label = label_info['label']
                f1 = label_info['f1']
                support = label_info['support']
                # Truncate for display
                display_label = label[:60] + "..." if len(label) > 63 else label
                print(f"  {i}. {display_label:<63} (F1: {f1:.3f}, Support: {support})")
    
    return all_language_results


def create_per_language_comparison_plots(all_results, output_dir, top_n=10):
    """Create visualization plots for per-language comparison."""
    plots_dir = os.path.join(output_dir, 'per_language_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    languages = sorted(all_results.keys())
    
    for label_type in ['narratives', 'subnarratives']:
        # Overall performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{label_type.capitalize()} - Per-Language Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Extract metrics
        f1_macro_scores = [all_results[lang][label_type]['f1_macro'] for lang in languages]
        f1_micro_scores = [all_results[lang][label_type]['f1_micro'] for lang in languages]
        f1_samples_scores = [all_results[lang][label_type]['f1_samples'] for lang in languages]
        total_labels = [all_results[lang][label_type]['total_labels'] for lang in languages]
        
        # F1 Macro
        bars1 = axes[0, 0].bar(languages, f1_macro_scores, color='skyblue')
        axes[0, 0].set_title('F1 Macro Score by Language')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim(0, 1)
        for bar, v in zip(bars1, f1_macro_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        # F1 Micro
        bars2 = axes[0, 1].bar(languages, f1_micro_scores, color='lightcoral')
        axes[0, 1].set_title('F1 Micro Score by Language')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_ylim(0, 1)
        for bar, v in zip(bars2, f1_micro_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        # F1 Samples
        bars3 = axes[1, 0].bar(languages, f1_samples_scores, color='lightgreen')
        axes[1, 0].set_title('F1 Samples Score by Language')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_ylim(0, 1)
        for bar, v in zip(bars3, f1_samples_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        # Total labels
        bars4 = axes[1, 1].bar(languages, total_labels, color='gold')
        axes[1, 1].set_title('Total Unique Labels by Language')
        axes[1, 1].set_ylabel('Number of Labels')
        for bar, v in zip(bars4, total_labels):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'{label_type}_per_language_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_file}")
        
        # Top performing labels heatmap for each language
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top labels across all languages
        all_top_labels = set()
        for lang in languages:
            top_labels = all_results[lang][label_type]['top_labels'][:top_n//2]  # Fewer for readability
            for label_info in top_labels:
                all_top_labels.add(label_info['label'])
        
        # Create matrix
        heatmap_data = []
        label_names = []
        
        for label in sorted(all_top_labels):
            row = []
            for lang in languages:
                # Find F1 score for this label in this language
                f1_score = 0
                for label_info in all_results[lang][label_type]['top_labels']:
                    if label_info['label'] == label:
                        f1_score = label_info['f1']
                        break
                row.append(f1_score)
            heatmap_data.append(row)
            # Truncate label names for display
            display_name = label[:50] + "..." if len(label) > 53 else label
            label_names.append(display_name)
        
        if heatmap_data:
            sns.heatmap(heatmap_data, 
                       xticklabels=languages, 
                       yticklabels=label_names,
                       annot=True, 
                       fmt='.3f', 
                       cmap='YlOrRd',
                       ax=ax)
            ax.set_title(f'{label_type.capitalize()} - Top Labels F1 Scores by Language')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, f'{label_type}_top_labels_heatmap.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_file}")


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
    parser.add_argument("--ground_truth", type=str,
                       help="Path to ground truth annotations file")
    parser.add_argument("--predictions", type=str,
                       help="Path to predictions file")
    parser.add_argument("--language", type=str,
                       help="Language code (e.g., EN, BG, HI, PT, RU)")
    parser.add_argument("--model_name", type=str, default="Unknown Model",
                       help="Name of the model being evaluated")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--experiment_name", type=str, default="evaluation",
                       help="Name of this evaluation experiment")
    
    # New argument for per-language analysis
    parser.add_argument("--analyze_results_dir", type=str,
                       help="Directory containing detailed results JSON files for per-language analysis")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top performing labels to show per language (default: 10)")
    
    args = parser.parse_args()
    
    # Check if we're doing per-language analysis of existing results
    if args.analyze_results_dir:
        print("=" * 80)
        print("ANALYZING EXISTING DETAILED RESULTS")
        print("=" * 80)
        
        if not os.path.exists(args.analyze_results_dir):
            print(f"Error: Results directory not found: {args.analyze_results_dir}")
            return
        
        # Analyze detailed results per language
        all_language_results = analyze_detailed_results_per_language(
            args.analyze_results_dir, 
            args.top_n
        )
        
        if all_language_results:
            # Create comparison plots
            print(f"\nGenerating per-language comparison plots...")
            create_per_language_comparison_plots(
                all_language_results, 
                args.output_dir, 
                args.top_n
            )
            
            # Save aggregated results
            aggregated_file = os.path.join(args.output_dir, 'per_language_analysis_summary.json')
            with open(aggregated_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_type': 'per_language_detailed_results',
                    'source_directory': args.analyze_results_dir,
                    'top_n': args.top_n,
                    'languages_analyzed': list(all_language_results.keys()),
                    'results': all_language_results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nAggregated analysis saved to: {aggregated_file}")
            
            # Create CSV summary
            summary_rows = []
            for lang in sorted(all_language_results.keys()):
                for label_type in ['narratives', 'subnarratives']:
                    data = all_language_results[lang][label_type]
                    summary_rows.append({
                        'Language': lang,
                        'Label_Type': label_type,
                        'F1_Macro': data['f1_macro'],
                        'F1_Micro': data['f1_micro'],
                        'F1_Samples': data['f1_samples'],
                        'Total_Labels': data['total_labels'],
                        'Total_Files': all_language_results[lang]['total_files'],
                        'Model': all_language_results[lang]['model']
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(args.output_dir, 'per_language_performance_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"Per-language summary CSV saved to: {summary_csv}")
        
        print(f"\n{'=' * 80}")
        print("PER-LANGUAGE ANALYSIS COMPLETE!")
        print(f"{'=' * 80}")
        return
    
    # Original single evaluation functionality
    if not args.ground_truth or not args.predictions or not args.language:
        parser.error("For single evaluation, --ground_truth, --predictions, and --language are required")
    
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