#!/usr/bin/env python3
"""
Generate comprehensive markdown analysis from Gemini 2.5 Flash detailed results
Focuses on F1 Samples as the primary metric
"""

import json
import numpy as np
import os
import argparse
from collections import defaultdict

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

def format_per_language_performance_table(language_results, label_type, top_n=10):
    """Generate per-language top N performance table focusing on F1 Samples"""
    markdown = f"\n## {label_type.capitalize()} Performance by Language (Top {top_n})\n\n"
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        results = data['results']
        model = data.get('model', 'Unknown')
        total_files = results.get('total_files', 0)
        
        if label_type not in results or 'per_label_metrics' not in results[label_type]:
            continue
        
        per_label_metrics = results[label_type]['per_label_metrics']
        
        # Get overall language performance
        f1_macro = results[label_type].get('f1_macro', 0)
        f1_micro = results[label_type].get('f1_micro', 0)
        f1_samples = results[label_type].get('f1_samples', 0)
        total_labels = results[label_type].get('total_labels', 0)
        
        # Sort labels by F1 score (using individual label F1 as proxy for F1 Samples contribution)
        sorted_labels = sorted(per_label_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        top_labels = sorted_labels[:top_n]
        
        markdown += f"### {lang_code} ({model})\n\n"
        markdown += f"**Overall Performance**: F1 Samples: {f1_samples:.4f} | F1 Macro: {f1_macro:.4f} | F1 Micro: {f1_micro:.4f} | Files: {total_files} | Labels: {total_labels}\n\n"
        
        markdown += "| Rank | Label | F1 Score | Precision | Recall | Support | TP | FP | FN | TN |\n"
        markdown += "|------|-------|----------|-----------|--------|---------|----|----|----|----|"
        
        for i, (label, metrics) in enumerate(top_labels, 1):
            f1 = metrics.get('f1', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            support = metrics.get('support', 0)
            tp = metrics.get('tp', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            tn = metrics.get('tn', 0)
            
            # Truncate long labels for display
            display_label = label[:60] + "..." if len(label) > 63 else label
            
            markdown += f"\n| {i:2d} | {display_label} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {support:3d} | {tp:2d} | {fp:2d} | {fn:2d} | {tn:2d} |"
        
        markdown += "\n\n"
    
    return markdown

def format_language_comparison_table(language_results):
    """Generate language comparison table focusing on F1 Samples"""
    markdown = "\n## Language Performance Comparison\n\n"
    markdown += "| Language | Model | Files | Narratives F1 Samples | Subnarratives F1 Samples | Narratives Labels | Subnarratives Labels |\n"
    markdown += "|----------|-------|-------|----------------------|--------------------------|-------------------|----------------------|\n"
    
    # Collect performance data
    performance_data = []
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        results = data['results']
        model = data.get('model', 'Unknown')
        total_files = results.get('total_files', 0)
        
        narratives_f1_samples = results.get('narratives', {}).get('f1_samples', 0)
        subnarratives_f1_samples = results.get('subnarratives', {}).get('f1_samples', 0)
        narratives_labels = results.get('narratives', {}).get('total_labels', 0)
        subnarratives_labels = results.get('subnarratives', {}).get('total_labels', 0)
        
        performance_data.append({
            'language': lang_code,
            'model': model,
            'files': total_files,
            'narratives_f1_samples': narratives_f1_samples,
            'subnarratives_f1_samples': subnarratives_f1_samples,
            'narratives_labels': narratives_labels,
            'subnarratives_labels': subnarratives_labels
        })
    
    # Sort by combined F1 Samples performance
    performance_data.sort(key=lambda x: (x['narratives_f1_samples'] + x['subnarratives_f1_samples']), reverse=True)
    
    for perf in performance_data:
        markdown += f"| {perf['language']} | {perf['model']} | {perf['files']} | {perf['narratives_f1_samples']:.4f} | {perf['subnarratives_f1_samples']:.4f} | {perf['narratives_labels']} | {perf['subnarratives_labels']} |\n"
    
    return markdown, performance_data

def generate_language_insights(performance_data):
    """Generate insights about language performance"""
    markdown = "\n## Performance Insights\n\n"
    
    # Best performing language overall
    best_overall = performance_data[0]
    worst_overall = performance_data[-1]
    
    markdown += f"### Overall Performance Rankings\n\n"
    markdown += f"**Best Overall Performance**: {best_overall['language']} (Combined F1 Samples: {best_overall['narratives_f1_samples'] + best_overall['subnarratives_f1_samples']:.4f})\n\n"
    markdown += f"**Lowest Overall Performance**: {worst_overall['language']} (Combined F1 Samples: {worst_overall['narratives_f1_samples'] + worst_overall['subnarratives_f1_samples']:.4f})\n\n"
    
    # Best for narratives
    best_narratives = max(performance_data, key=lambda x: x['narratives_f1_samples'])
    markdown += f"**Best Narratives Performance**: {best_narratives['language']} (F1 Samples: {best_narratives['narratives_f1_samples']:.4f})\n\n"
    
    # Best for subnarratives
    best_subnarratives = max(performance_data, key=lambda x: x['subnarratives_f1_samples'])
    markdown += f"**Best Subnarratives Performance**: {best_subnarratives['language']} (F1 Samples: {best_subnarratives['subnarratives_f1_samples']:.4f})\n\n"
    
    # Performance distribution
    narratives_scores = [p['narratives_f1_samples'] for p in performance_data]
    subnarratives_scores = [p['subnarratives_f1_samples'] for p in performance_data]
    
    markdown += f"### Performance Statistics\n\n"
    markdown += f"**Narratives F1 Samples**:\n"
    markdown += f"- Average: {np.mean(narratives_scores):.4f}\n"
    markdown += f"- Range: {min(narratives_scores):.4f} - {max(narratives_scores):.4f}\n"
    markdown += f"- Standard Deviation: {np.std(narratives_scores):.4f}\n\n"
    
    markdown += f"**Subnarratives F1 Samples**:\n"
    markdown += f"- Average: {np.mean(subnarratives_scores):.4f}\n"
    markdown += f"- Range: {min(subnarratives_scores):.4f} - {max(subnarratives_scores):.4f}\n"
    markdown += f"- Standard Deviation: {np.std(subnarratives_scores):.4f}\n\n"
    
    # Performance gaps
    performance_gaps = [(p['narratives_f1_samples'] - p['subnarratives_f1_samples']) for p in performance_data]
    avg_gap = np.mean(performance_gaps)
    
    markdown += f"### Narratives vs Subnarratives Performance Gap\n\n"
    markdown += f"- Average gap (Narratives - Subnarratives): {avg_gap:.4f}\n"
    markdown += f"- Languages with smallest gap: {min(performance_data, key=lambda x: abs(x['narratives_f1_samples'] - x['subnarratives_f1_samples']))['language']}\n"
    markdown += f"- Languages with largest gap: {max(performance_data, key=lambda x: abs(x['narratives_f1_samples'] - x['subnarratives_f1_samples']))['language']}\n\n"
    
    return markdown

def main():
    parser = argparse.ArgumentParser(description="Generate Gemini 2.5 Flash results analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing detailed results JSON files")
    parser.add_argument("--output_file", type=str, default="gemini_flash_analysis.md",
                       help="Output markdown file name")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top performing labels to show per language")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    print(f"Loading Gemini 2.5 Flash results from: {args.results_dir}")
    
    # Load all language results
    language_results = load_detailed_results(args.results_dir)
    
    if not language_results:
        print("No detailed results files found!")
        return
    
    print(f"Found results for languages: {', '.join(sorted(language_results.keys()))}")
    
    # Get experiment info
    first_result = next(iter(language_results.values()))
    experiment_name = first_result.get('experiment_name', 'Gemini 2.5 Flash Analysis')
    model_name = first_result.get('model', 'Google Gemini 2.5 Flash')
    
    print("Generating analysis...")
    
    # Generate language comparison
    language_comparison, performance_data = format_language_comparison_table(language_results)
    
    # Generate per-language analysis
    narratives_analysis = format_per_language_performance_table(language_results, 'narratives', args.top_n)
    subnarratives_analysis = format_per_language_performance_table(language_results, 'subnarratives', args.top_n)
    
    # Generate insights
    insights = generate_language_insights(performance_data)
    
    # Create comprehensive markdown
    markdown_output = f"""# Gemini 2.5 Flash Results Analysis

**Experiment**: {experiment_name}  
**Model**: {model_name}  
**Languages Analyzed**: {', '.join(sorted(language_results.keys()))}  
**Primary Metric**: F1 Samples  

This analysis summarizes the performance of Gemini 2.5 Flash across different languages for hierarchical text classification.

{language_comparison}

{insights}

{narratives_analysis}

{subnarratives_analysis}

## Summary

This analysis provides a comprehensive view of Gemini 2.5 Flash performance across languages, focusing on F1 Samples as the primary evaluation metric. The results show language-specific performance patterns that can inform deployment strategies and model optimization efforts.
"""
    
    # Save to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_output)
    
    print(f"\nAnalysis saved to: {args.output_file}")
    print(f"Languages analyzed: {sorted(language_results.keys())}")
    print(f"Top {args.top_n} labels shown per language")

if __name__ == "__main__":
    main()