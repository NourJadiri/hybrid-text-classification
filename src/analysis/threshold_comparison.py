#!/usr/bin/env python3
"""
Threshold Comparison Analysis Script

This script compares performance metrics across different classification thresholds.
It reads the analysis results from different threshold folders and creates a comprehensive
comparison showing how threshold affects model performance.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

def load_threshold_results(analysis_dir, thresholds):
    """
    Load performance results for all thresholds.
    
    Args:
        analysis_dir (str): Base directory containing threshold folders
        thresholds (list): List of threshold values to compare
    
    Returns:
        dict: Nested dictionary with threshold -> metric -> value structure
    """
    results = {}
    
    for threshold in thresholds:
        threshold_dir = os.path.join(analysis_dir, str(threshold))
        summary_file = os.path.join(threshold_dir, 'performance_summary.csv')
        
        if os.path.exists(summary_file):
            print(f"Loading results for threshold {threshold}...")
            df = pd.read_csv(summary_file)
            
            # Calculate overall metrics for this threshold
            results[threshold] = {
                'narratives': {
                    'f1_macro': df[df['Label_Type'] == 'narratives']['F1_Macro'].mean(),
                    'f1_micro': df[df['Label_Type'] == 'narratives']['F1_Micro'].mean(),
                    'f1_samples': df[df['Label_Type'] == 'narratives']['F1_Samples'].mean(),
                },
                'subnarratives': {
                    'f1_macro': df[df['Label_Type'] == 'subnarratives']['F1_Macro'].mean(),
                    'f1_micro': df[df['Label_Type'] == 'subnarratives']['F1_Micro'].mean(),
                    'f1_samples': df[df['Label_Type'] == 'subnarratives']['F1_Samples'].mean(),
                },
                'by_language': {}
            }
            
            # Store per-language results
            for _, row in df.iterrows():
                lang = row['Language']
                label_type = row['Label_Type']
                
                if lang not in results[threshold]['by_language']:
                    results[threshold]['by_language'][lang] = {}
                
                results[threshold]['by_language'][lang][label_type] = {
                    'f1_macro': row['F1_Macro'],
                    'f1_micro': row['F1_Micro'],
                    'f1_samples': row['F1_Samples']
                }
        else:
            print(f"Warning: Summary file not found for threshold {threshold}: {summary_file}")
    
    return results

def create_comparison_plots(results, output_dir):
    """Create comprehensive comparison plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Overall F1 Macro comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    thresholds = sorted(results.keys(), key=float)
    
    # Extract metrics for plotting
    narratives_f1_macro = [results[t]['narratives']['f1_macro'] for t in thresholds]
    narratives_f1_micro = [results[t]['narratives']['f1_micro'] for t in thresholds]
    subnarratives_f1_macro = [results[t]['subnarratives']['f1_macro'] for t in thresholds]
    subnarratives_f1_micro = [results[t]['subnarratives']['f1_micro'] for t in thresholds]
    
    # Plot 1: Narratives F1 Macro
    ax1.plot(thresholds, narratives_f1_macro, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_title('Narratives - F1 Macro Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('F1 Macro Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(narratives_f1_macro) * 1.1)
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(thresholds, narratives_f1_macro)):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Plot 2: Narratives F1 Micro
    ax2.plot(thresholds, narratives_f1_micro, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_title('Narratives - F1 Micro Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('F1 Micro Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(narratives_f1_micro) * 1.1)
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(thresholds, narratives_f1_micro)):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Plot 3: Subnarratives F1 Macro
    ax3.plot(thresholds, subnarratives_f1_macro, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_title('Subnarratives - F1 Macro Score vs Threshold', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Classification Threshold')
    ax3.set_ylabel('F1 Macro Score')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(subnarratives_f1_macro) * 1.1)
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(thresholds, subnarratives_f1_macro)):
        ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Plot 4: Subnarratives F1 Micro
    ax4.plot(thresholds, subnarratives_f1_micro, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    ax4.set_title('Subnarratives - F1 Micro Score vs Threshold', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Classification Threshold')
    ax4.set_ylabel('F1 Micro Score')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(subnarratives_f1_micro) * 1.1)
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(thresholds, subnarratives_f1_micro)):
        ax4.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_comparison_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-language comparison heatmap
    languages = ['BG', 'EN', 'HI', 'PT', 'RU']
    
    # Create heatmaps for narratives and subnarratives F1 Macro
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Narratives heatmap data
    narratives_data = []
    subnarratives_data = []
    
    for lang in languages:
        narratives_row = []
        subnarratives_row = []
        for threshold in thresholds:
            if lang in results[threshold]['by_language']:
                narratives_row.append(results[threshold]['by_language'][lang]['narratives']['f1_macro'])
                subnarratives_row.append(results[threshold]['by_language'][lang]['subnarratives']['f1_macro'])
            else:
                narratives_row.append(0)
                subnarratives_row.append(0)
        narratives_data.append(narratives_row)
        subnarratives_data.append(subnarratives_row)
    
    # Plot narratives heatmap
    sns.heatmap(narratives_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=thresholds, yticklabels=languages, ax=ax1, cbar_kws={'label': 'F1 Macro Score'})
    ax1.set_title('Narratives F1 Macro by Language and Threshold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('Language')
    
    # Plot subnarratives heatmap
    sns.heatmap(subnarratives_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=thresholds, yticklabels=languages, ax=ax2, cbar_kws={'label': 'F1 Macro Score'})
    ax2.set_title('Subnarratives F1 Macro by Language and Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('Language')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_comparison_by_language.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results, output_dir):
    """Create summary table with all metrics."""
    
    # Create comprehensive summary table
    summary_data = []
    
    thresholds = sorted(results.keys(), key=float)
    
    for threshold in thresholds:
        # Overall metrics
        summary_data.append({
            'Threshold': threshold,
            'Category': 'Overall',
            'Label_Type': 'Narratives',
            'F1_Macro': results[threshold]['narratives']['f1_macro'],
            'F1_Micro': results[threshold]['narratives']['f1_micro'],
            'F1_Samples': results[threshold]['narratives']['f1_samples']
        })
        summary_data.append({
            'Threshold': threshold,
            'Category': 'Overall',
            'Label_Type': 'Subnarratives',
            'F1_Macro': results[threshold]['subnarratives']['f1_macro'],
            'F1_Micro': results[threshold]['subnarratives']['f1_micro'],
            'F1_Samples': results[threshold]['subnarratives']['f1_samples']
        })
        
        # Per-language metrics
        for lang in ['BG', 'EN', 'HI', 'PT', 'RU']:
            if lang in results[threshold]['by_language']:
                for label_type in ['narratives', 'subnarratives']:
                    summary_data.append({
                        'Threshold': threshold,
                        'Category': lang,
                        'Label_Type': label_type,
                        'F1_Macro': results[threshold]['by_language'][lang][label_type]['f1_macro'],
                        'F1_Micro': results[threshold]['by_language'][lang][label_type]['f1_micro'],
                        'F1_Samples': results[threshold]['by_language'][lang][label_type]['f1_samples']
                    })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'threshold_comparison_summary.csv'), index=False)
    
    # Create best performance summary
    best_results = []
    
    # Find best thresholds for each metric
    for label_type in ['Narratives', 'Subnarratives']:
        overall_data = df[(df['Category'] == 'Overall') & (df['Label_Type'] == label_type)]
        
        for metric in ['F1_Macro', 'F1_Micro', 'F1_Samples']:
            best_row = overall_data.loc[overall_data[metric].idxmax()]
            best_results.append({
                'Label_Type': label_type,
                'Metric': metric,
                'Best_Threshold': best_row['Threshold'],
                'Best_Score': best_row[metric],
                'Improvement_vs_0.65': best_row[metric] - df[(df['Threshold'] == '0.65') & 
                                                           (df['Category'] == 'Overall') & 
                                                           (df['Label_Type'] == label_type)][metric].iloc[0]
            })
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(os.path.join(output_dir, 'best_thresholds_summary.csv'), index=False)
    
    return df, best_df

def print_summary_report(results, best_df):
    """Print a comprehensive summary report."""
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON ANALYSIS - SUMMARY REPORT")
    print("="*80)
    
    thresholds = sorted(results.keys(), key=float)
    
    print(f"\nAnalyzed Thresholds: {', '.join(thresholds)}")
    print(f"Training Threshold: 0.65")
    
    print("\n" + "-"*60)
    print("OVERALL PERFORMANCE COMPARISON")
    print("-"*60)
    
    # Compare overall metrics
    for label_type in ['narratives', 'subnarratives']:
        print(f"\n{label_type.capitalize()}:")
        print(f"{'Threshold':<12} {'F1 Macro':<12} {'F1 Micro':<12} {'F1 Samples':<12}")
        print("-" * 50)
        
        for threshold in thresholds:
            metrics = results[threshold][label_type]
            print(f"{threshold:<12} {metrics['f1_macro']:<12.4f} {metrics['f1_micro']:<12.4f} {metrics['f1_samples']:<12.4f}")
    
    print("\n" + "-"*60)
    print("BEST PERFORMING THRESHOLDS")
    print("-"*60)
    
    for _, row in best_df.iterrows():
        improvement = row['Improvement_vs_0.65']
        direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"{row['Label_Type']} - {row['Metric']}: Threshold {row['Best_Threshold']} "
              f"(Score: {row['Best_Score']:.4f}, {direction} {improvement:+.4f} vs 0.65)")
    
    print("\n" + "-"*60)
    print("KEY INSIGHTS")
    print("-"*60)
    
    # Calculate some insights
    best_overall_narratives = max(thresholds, key=lambda t: results[t]['narratives']['f1_macro'])
    best_overall_subnarratives = max(thresholds, key=lambda t: results[t]['subnarratives']['f1_macro'])
    
    print(f"• Best overall threshold for narratives: {best_overall_narratives}")
    print(f"• Best overall threshold for subnarratives: {best_overall_subnarratives}")
    
    # Compare to training threshold
    training_perf_narr = results['0.65']['narratives']['f1_macro']
    training_perf_sub = results['0.65']['subnarratives']['f1_macro']
    
    better_narr = [t for t in thresholds if results[t]['narratives']['f1_macro'] > training_perf_narr]
    better_sub = [t for t in thresholds if results[t]['subnarratives']['f1_macro'] > training_perf_sub]
    
    if better_narr:
        print(f"• Thresholds better than training for narratives: {', '.join(better_narr)}")
    else:
        print("• No threshold outperformed training threshold for narratives")
        
    if better_sub:
        print(f"• Thresholds better than training for subnarratives: {', '.join(better_sub)}")
    else:
        print("• No threshold outperformed training threshold for subnarratives")

def main():
    parser = argparse.ArgumentParser(description="Compare performance across different classification thresholds")
    parser.add_argument("--analysis_dir", type=str, default="../../results/analysis",
                        help="Directory containing threshold-specific analysis results")
    parser.add_argument("--output_dir", type=str, default="../../results/threshold_comparison",
                        help="Directory to save comparison results")
    parser.add_argument("--thresholds", nargs='+', default=['0.5', '0.65', '0.7', '0.8'],
                        help="List of thresholds to compare")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading threshold analysis results...")
    results = load_threshold_results(args.analysis_dir, args.thresholds)
    
    if not results:
        print("Error: No results found. Make sure the analysis has been run for the specified thresholds.")
        return
    
    print("Creating comparison plots...")
    create_comparison_plots(results, args.output_dir)
    
    print("Creating summary tables...")
    summary_df, best_df = create_summary_table(results, args.output_dir)
    
    print("Generating summary report...")
    print_summary_report(results, best_df)
    
    print(f"\n{'='*80}")
    print("THRESHOLD COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved in: {args.output_dir}")
    print("Generated files:")
    print("• threshold_comparison_overview.png - Overall performance trends")
    print("• threshold_comparison_by_language.png - Per-language heatmaps")
    print("• threshold_comparison_summary.csv - Complete results table")
    print("• best_thresholds_summary.csv - Best threshold for each metric")

if __name__ == "__main__":
    main()