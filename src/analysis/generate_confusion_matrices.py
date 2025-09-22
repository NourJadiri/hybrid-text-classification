#!/usr/bin/env python3
"""
Generate comprehensive confusion matrices for all languages, narratives, and subnarratives
Creates a detailed markdown file with confusion matrix for every label in every language
"""

import json
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

def generate_language_confusion_matrices(language_results, label_type, lang_code, data):
    """Generate confusion matrices tables for all labels in a specific language"""
    results = data['results']
    
    if label_type not in results or 'per_label_metrics' not in results[label_type]:
        return f"\n### {lang_code} - {label_type.capitalize()}\n\n*No {label_type} data available for this language.*\n\n"
    
    per_label_metrics = results[label_type]['per_label_metrics']
    model = data.get('model', 'Unknown')
    total_files = results.get('total_files', 0)
    
    # Overall metrics for this language and label type
    f1_macro = results[label_type].get('f1_macro', 0)
    f1_micro = results[label_type].get('f1_micro', 0)
    f1_samples = results[label_type].get('f1_samples', 0)
    total_labels = results[label_type].get('total_labels', 0)
    
    markdown = f"\n### {lang_code} - {label_type.capitalize()}\n\n"
    markdown += f"**Language**: {lang_code}  \n"
    markdown += f"**Model**: {model}  \n"
    markdown += f"**Total Files**: {total_files}  \n"
    markdown += f"**Total Labels**: {total_labels}  \n"
    markdown += f"**F1 Macro**: {f1_macro:.4f}  \n"
    markdown += f"**F1 Micro**: {f1_micro:.4f}  \n"
    markdown += f"**F1 Samples**: {f1_samples:.4f}  \n\n"
    
    # Sort labels by F1 score for better organization
    sorted_labels = sorted(per_label_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # Create comprehensive table
    markdown += f"#### Confusion Matrix Summary Table ({len(sorted_labels)} labels)\n\n"
    markdown += "| Label | F1 | Precision | Recall | Support | TP | FP | FN | TN |\n"
    markdown += "|-------|----|-----------|--------|---------|----|----|----|----|\n"
    
    for label, metrics in sorted_labels:
        tp = metrics.get('tp', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        tn = metrics.get('tn', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        support = metrics.get('support', 0)
        
        # Truncate long labels for table display
        display_label = label[:50] + "..." if len(label) > 53 else label
        
        markdown += f"| {display_label} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {support:3d} | {tp:2d} | {fp:2d} | {fn:2d} | {tn:3d} |\n"
    
    markdown += "\n**Legend:**\n"
    markdown += "- **TP**: True Positives (correctly predicted positive)\n"
    markdown += "- **FP**: False Positives (incorrectly predicted positive)\n"
    markdown += "- **FN**: False Negatives (incorrectly predicted negative)\n"
    markdown += "- **TN**: True Negatives (correctly predicted negative)\n\n"
    
    return markdown

def generate_summary_statistics(language_results):
    """Generate summary statistics across all languages"""
    summary = "\n## Summary Statistics\n\n"
    
    # Language performance summary
    summary += "### Performance Overview by Language\n\n"
    summary += "| Language | Model | Files | Narratives F1 | Subnarratives F1 | Narratives Labels | Subnarratives Labels |\n"
    summary += "|----------|-------|-------|---------------|------------------|-------------------|----------------------|\n"
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        results = data['results']
        model = data.get('model', 'Unknown')
        total_files = results.get('total_files', 0)
        
        narr_f1 = results.get('narratives', {}).get('f1_samples', 0)
        subnarr_f1 = results.get('subnarratives', {}).get('f1_samples', 0)
        narr_labels = results.get('narratives', {}).get('total_labels', 0)
        subnarr_labels = results.get('subnarratives', {}).get('total_labels', 0)
        
        summary += f"| {lang_code} | {model} | {total_files} | {narr_f1:.4f} | {subnarr_f1:.4f} | {narr_labels} | {subnarr_labels} |\n"
    
    # Label count statistics
    summary += "\n### Label Distribution Statistics\n\n"
    
    for label_type in ['narratives', 'subnarratives']:
        total_unique_labels = set()
        total_matrices = 0
        
        for lang_code, data in language_results.items():
            results = data['results']
            if label_type in results and 'per_label_metrics' in results[label_type]:
                per_label_metrics = results[label_type]['per_label_metrics']
                for label in per_label_metrics.keys():
                    total_unique_labels.add(label)
                total_matrices += len(per_label_metrics)
        
        summary += f"**{label_type.capitalize()}:**\n"
        summary += f"- Total unique labels across all languages: {len(total_unique_labels)}\n"
        summary += f"- Total confusion matrices generated: {total_matrices}\n\n"
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive confusion matrices for Gemini 2.5 Flash results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing detailed results JSON files")
    parser.add_argument("--output_file", type=str, default="gemini_confusion_matrices.md",
                       help="Output markdown file name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Set output file in the same directory as results
    output_path = os.path.join(args.results_dir, args.output_file)
    
    print(f"Loading Gemini 2.5 Flash results from: {args.results_dir}")
    
    # Load all language results
    language_results = load_detailed_results(args.results_dir)
    
    if not language_results:
        print("No detailed results files found!")
        return
    
    print(f"Found results for languages: {', '.join(sorted(language_results.keys()))}")
    print(f"Generating comprehensive confusion matrices...")
    
    # Get experiment info
    first_result = next(iter(language_results.values()))
    experiment_name = first_result.get('experiment_name', 'Gemini 2.5 Flash Analysis')
    model_name = first_result.get('model', 'Google Gemini 2.5 Flash')
    
    # Generate markdown content
    markdown_content = f"""# Comprehensive Confusion Matrices - Gemini 2.5 Flash Results

**Experiment**: {experiment_name}  
**Model**: {model_name}  
**Languages Analyzed**: {', '.join(sorted(language_results.keys()))}  
**Analysis Type**: Per-Label Confusion Matrices  

This document provides detailed confusion matrices for every narrative and subnarrative label in every language analyzed by Gemini 2.5 Flash.

"""
    
    # Add summary statistics
    markdown_content += generate_summary_statistics(language_results)
    
    # Generate detailed confusion matrices for each language and label type
    markdown_content += "\n---\n\n# Detailed Confusion Matrices\n\n"
    
    # Process narratives first
    markdown_content += "\n## Narratives Confusion Matrices\n\n"
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        markdown_content += generate_language_confusion_matrices(language_results, 'narratives', lang_code, data)
    
    # Then process subnarratives
    markdown_content += "\n## Subnarratives Confusion Matrices\n\n"
    
    for lang_code in sorted(language_results.keys()):
        data = language_results[lang_code]
        markdown_content += generate_language_confusion_matrices(language_results, 'subnarratives', lang_code, data)
    
    # Add footer
    markdown_content += f"""
---

## Notes

- **TP (True Positives)**: Correctly identified positive cases
- **FP (False Positives)**: Incorrectly identified positive cases (Type I error)
- **TN (True Negatives)**: Correctly identified negative cases
- **FN (False Negatives)**: Incorrectly identified negative cases (Type II error)

- **Precision**: TP / (TP + FP) - How many selected items are relevant
- **Recall**: TP / (TP + FN) - How many relevant items are selected
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall

**Generated for**: {experiment_name}  
**Model**: {model_name}  
**Languages**: {', '.join(sorted(language_results.keys()))}
"""
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Count total matrices generated
    total_matrices = 0
    for lang_code, data in language_results.items():
        results = data['results']
        for label_type in ['narratives', 'subnarratives']:
            if label_type in results and 'per_label_metrics' in results[label_type]:
                total_matrices += len(results[label_type]['per_label_metrics'])
    
    print(f"\n{'=' * 70}")
    print("CONFUSION MATRICES GENERATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Comprehensive confusion matrices saved to: {output_path}")
    print(f"Languages analyzed: {sorted(language_results.keys())}")
    print(f"Total confusion matrices generated: {total_matrices}")
    print(f"  - Narratives matrices: {sum([len(data['results'].get('narratives', {}).get('per_label_metrics', {})) for data in language_results.values()])}")
    print(f"  - Subnarratives matrices: {sum([len(data['results'].get('subnarratives', {}).get('per_label_metrics', {})) for data in language_results.values()])}")

if __name__ == "__main__":
    main()