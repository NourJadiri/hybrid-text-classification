#!/usr/bin/env python3
"""
Generate specific plots for Gemini 2.5 Flash results per language
Each plot is saved in its own file, focusing on F1 Samples as primary metric
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_detailed_results(results_dir):
    """Load all detailed results JSON files from directory"""
    language_results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_detailed_results.json'):
            language_code = filename.split('_')[0].upper()
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                language_results[language_code] = data
    
    return language_results

def plot_top_performing_labels(language_results, label_type, top_n, output_dir):
    """Generate top performing narratives/subnarratives plots per language"""
    
    for lang_code, data in language_results.items():
        results = data['results']
        
        if label_type not in results or 'per_label_metrics' not in results[label_type]:
            continue
        
        per_label_metrics = results[label_type]['per_label_metrics']
        
        # Sort by F1 score and get top N
        sorted_labels = sorted(per_label_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        top_labels = sorted_labels[:top_n]
        
        if not top_labels:
            continue
        
        # Prepare data for plotting
        labels = []
        f1_scores = []
        
        for label, metrics in top_labels:
            # Truncate long labels for display
            display_label = label[:40] + "..." if len(label) > 43 else label
            labels.append(display_label)
            f1_scores.append(metrics['f1'])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(labels)), f1_scores, color='steelblue', alpha=0.8)
        
        # Customize plot
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('F1 Score', fontsize=12, fontweight='bold')
        plt.title(f'{lang_code} - Top {len(top_labels)} {label_type.capitalize()} by F1 Score', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # Set x-axis limits
        plt.xlim(0, max(1.0, max(f1_scores) + 0.1))
        
        # Add grid for better readability
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Invert y-axis so highest scores are at top
        plt.gca().invert_yaxis()
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        filename = f'{lang_code}_top_{label_type}_f1_scores.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filepath}")

def plot_f1_samples_by_language(language_results, output_dir):
    """Generate F1 Samples bar chart for each language"""
    
    # Prepare data
    languages = []
    narratives_f1_samples = []
    subnarratives_f1_samples = []
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        results = data['results']
        
        languages.append(lang_code)
        narratives_f1_samples.append(results.get('narratives', {}).get('f1_samples', 0))
        subnarratives_f1_samples.append(results.get('subnarratives', {}).get('f1_samples', 0))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(languages))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, narratives_f1_samples, width, 
                   label='Narratives', color='skyblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, subnarratives_f1_samples, width, 
                   label='Subnarratives', color='lightcoral', alpha=0.8)
    
    # Customize plot
    plt.xlabel('Language', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Samples Score', fontsize=12, fontweight='bold')
    plt.title('F1 Samples Scores by Language', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, languages)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    filename = 'f1_samples_by_language.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")

def create_summary_stats_plots(language_results, output_dir):
    """Create additional summary statistics plots"""
    
    # Language performance comparison (separate plot)
    languages = sorted(language_results.keys())
    
    # Overall F1 Samples comparison
    plt.figure(figsize=(10, 6))
    
    narratives_scores = []
    subnarratives_scores = []
    combined_scores = []
    
    for lang in languages:
        data = language_results[lang]
        results = data['results']
        
        narr_f1 = results.get('narratives', {}).get('f1_samples', 0)
        subnarr_f1 = results.get('subnarratives', {}).get('f1_samples', 0)
        
        narratives_scores.append(narr_f1)
        subnarratives_scores.append(subnarr_f1)
        combined_scores.append((narr_f1 + subnarr_f1) / 2)
    
    x = np.arange(len(languages))
    bars = plt.bar(x, combined_scores, color='mediumseagreen', alpha=0.8)
    
    plt.xlabel('Language', fontsize=12, fontweight='bold')
    plt.ylabel('Average F1 Samples Score', fontsize=12, fontweight='bold')
    plt.title('Combined Performance (Average F1 Samples) by Language', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, languages)
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, combined_scores):
        plt.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    filename = 'combined_performance_by_language.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate Gemini 2.5 Flash visualization plots per language")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing detailed results JSON files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (defaults to results_dir/plots)")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top performing labels to show")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Set output directory to results_dir/plots if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    
    # Create output directory with proper structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Gemini 2.5 Flash results from: {args.results_dir}")
    
    # Load all language results
    language_results = load_detailed_results(args.results_dir)
    
    if not language_results:
        print("No detailed results files found!")
        return
    
    print(f"Found results for languages: {', '.join(sorted(language_results.keys()))}")
    print(f"Generating plots in: {args.output_dir}")
    print()
    
    # Generate plots
    print("1. Generating top performing narratives plots per language...")
    plot_top_performing_labels(language_results, 'narratives', args.top_n, args.output_dir)
    
    print("\n2. Generating top performing subnarratives plots per language...")
    plot_top_performing_labels(language_results, 'subnarratives', args.top_n, args.output_dir)
    
    print("\n3. Generating F1 Samples comparison plot...")
    plot_f1_samples_by_language(language_results, args.output_dir)
    
    print("\n4. Generating summary statistics plots...")
    create_summary_stats_plots(language_results, args.output_dir)
    
    print(f"\n{'=' * 60}")
    print("PLOT GENERATION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"All plots saved in: {args.output_dir}")
    print(f"Languages analyzed: {sorted(language_results.keys())}")
    print(f"Top {args.top_n} labels shown per language")

if __name__ == "__main__":
    main()