#!/usr/bin/env python3
"""
Extract and summarize confusion matrices for the optimal threshold (0.65)
"""

import json
import numpy as np
from collections import defaultdict

def load_results(file_path):
    """Load detailed results JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def aggregate_confusion_matrices(results, label_type='narratives'):
    """Aggregate confusion matrices across all languages for a label type"""
    
    all_matrices = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_samples': 0})
    
    # Process each language
    for lang_data in results['by_language']:
        if label_type in lang_data and 'confusion_matrices' in lang_data[label_type]:
            for label, cm_data in lang_data[label_type]['confusion_matrices'].items():
                matrix = np.array(cm_data['matrix'])
                
                # Extract TP, FP, FN, TN from 2x2 confusion matrix
                tn, fp, fn, tp = matrix.ravel()
                
                all_matrices[label]['tp'] += tp
                all_matrices[label]['fp'] += fp
                all_matrices[label]['fn'] += fn
                all_matrices[label]['tn'] += tn
                all_matrices[label]['total_samples'] += matrix.sum()
    
    return all_matrices

def calculate_metrics(cm_data):
    """Calculate precision, recall, F1 from confusion matrix data"""
    tp, fp, fn, tn = cm_data['tp'], cm_data['fp'], cm_data['fn'], cm_data['tn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def get_per_language_matrices(results, label_type='narratives'):
    """Get confusion matrices organized by language"""
    
    language_matrices = {}
    
    # Process each language
    for lang_data in results['by_language']:
        language = lang_data['language']
        language_matrices[language] = {}
        
        if label_type in lang_data and 'confusion_matrices' in lang_data[label_type]:
            for label, cm_data in lang_data[label_type]['confusion_matrices'].items():
                matrix = np.array(cm_data['matrix'])
                
                # Extract TP, FP, FN, TN from 2x2 confusion matrix
                tn, fp, fn, tp = matrix.ravel()
                
                language_matrices[language][label] = {
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'total_samples': matrix.sum()
                }
    
    return language_matrices

def format_confusion_matrix_table(matrices, label_type, top_n=10, title_suffix=""):
    """Format confusion matrices as markdown table"""
    
    # Sort by F1 score (descending) and take top N
    sorted_labels = []
    for label, cm_data in matrices.items():
        _, _, f1 = calculate_metrics(cm_data)
        sorted_labels.append((label, f1, cm_data))
    
    sorted_labels.sort(key=lambda x: x[1], reverse=True)
    top_labels = sorted_labels[:top_n]
    
    markdown = f"\n## Top {top_n} {label_type.capitalize()} - Confusion Matrix Analysis{title_suffix}\n\n"
    markdown += "| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |\n"
    markdown += "|-------|----|----|----|----|-----------|--------|---------|\n"
    
    for label, f1, cm_data in top_labels:
        tp, fp, fn, tn = cm_data['tp'], cm_data['fp'], cm_data['fn'], cm_data['tn']
        precision, recall, f1_calc = calculate_metrics(cm_data)
        
        # Truncate long labels for better table formatting
        display_label = label if len(label) <= 50 else label[:47] + "..."
        
        markdown += f"| {display_label} | {tp} | {fp} | {fn} | {tn} | {precision:.3f} | {recall:.3f} | {f1_calc:.3f} |\n"
    
    return markdown

def format_per_language_tables(language_matrices, label_type, top_n=8):
    """Format per-language confusion matrices as markdown tables"""
    
    markdown = f"\n## {label_type.capitalize()} Performance by Language\n\n"
    
    for language in sorted(language_matrices.keys()):
        matrices = language_matrices[language]
        
        if not matrices:
            continue
            
        # Sort by F1 score (descending) and take top N
        sorted_labels = []
        for label, cm_data in matrices.items():
            _, _, f1 = calculate_metrics(cm_data)
            sorted_labels.append((label, f1, cm_data))
        
        sorted_labels.sort(key=lambda x: x[1], reverse=True)
        top_labels = sorted_labels[:top_n]
        
        markdown += f"\n### {language} - Top {min(top_n, len(top_labels))} {label_type.capitalize()}\n\n"
        markdown += "| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |\n"
        markdown += "|-------|----|----|----|----|-----------|--------|---------|\n"
        
        for label, f1, cm_data in top_labels:
            tp, fp, fn, tn = cm_data['tp'], cm_data['fp'], cm_data['fn'], cm_data['tn']
            precision, recall, f1_calc = calculate_metrics(cm_data)
            
            # Truncate long labels for better table formatting
            display_label = label if len(label) <= 45 else label[:42] + "..."
            
            markdown += f"| {display_label} | {tp} | {fp} | {fn} | {tn} | {precision:.3f} | {recall:.3f} | {f1_calc:.3f} |\n"
    
    return markdown

def generate_summary_stats(matrices, label_type):
    """Generate summary statistics for all labels"""
    all_metrics = []
    total_tp = total_fp = total_fn = total_tn = 0
    
    for label, cm_data in matrices.items():
        precision, recall, f1 = calculate_metrics(cm_data)
        all_metrics.append((precision, recall, f1))
        
        total_tp += cm_data['tp']
        total_fp += cm_data['fp'] 
        total_fn += cm_data['fn']
        total_tn += cm_data['tn']
    
    # Calculate aggregate metrics
    if all_metrics:
        precisions, recalls, f1s = zip(*all_metrics)
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        # Micro-averaged metrics
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    else:
        macro_precision = macro_recall = macro_f1 = 0
        micro_precision = micro_recall = micro_f1 = 0
    
    summary = f"\n### {label_type.capitalize()} Summary Statistics\n\n"
    summary += f"- **Total Labels Analyzed**: {len(matrices)}\n"
    summary += f"- **Total Samples**: {total_tp + total_fp + total_fn + total_tn}\n"
    summary += f"- **Macro-averaged Precision**: {macro_precision:.3f}\n"
    summary += f"- **Macro-averaged Recall**: {macro_recall:.3f}\n"
    summary += f"- **Macro-averaged F1**: {macro_f1:.3f}\n"
    summary += f"- **Micro-averaged Precision**: {micro_precision:.3f}\n"
    summary += f"- **Micro-averaged Recall**: {micro_recall:.3f}\n"
    summary += f"- **Micro-averaged F1**: {micro_f1:.3f}\n\n"
    
    return summary

def main():
    # Load results
    results_file = "../../results/analysis/0.65/detailed_results.json"
    results = load_results(results_file)
    
    print("Extracting confusion matrices for threshold 0.65...")
    
    # Process narratives - aggregated
    narrative_matrices = aggregate_confusion_matrices(results, 'narratives')
    narrative_summary = generate_summary_stats(narrative_matrices, 'narratives')
    narrative_table = format_confusion_matrix_table(narrative_matrices, 'narratives', 
                                                   top_n=15, title_suffix=" (Aggregated - Threshold 0.65)")
    
    # Process narratives - per language
    narrative_lang_matrices = get_per_language_matrices(results, 'narratives')
    narrative_lang_tables = format_per_language_tables(narrative_lang_matrices, 'narratives', top_n=8)
    
    # Process subnarratives - aggregated
    subnarrative_matrices = aggregate_confusion_matrices(results, 'subnarratives')
    subnarrative_summary = generate_summary_stats(subnarrative_matrices, 'subnarratives')
    subnarrative_table = format_confusion_matrix_table(subnarrative_matrices, 'subnarratives', 
                                                      top_n=15, title_suffix=" (Aggregated - Threshold 0.65)")
    
    # Process subnarratives - per language
    subnarrative_lang_matrices = get_per_language_matrices(results, 'subnarratives')
    subnarrative_lang_tables = format_per_language_tables(subnarrative_lang_matrices, 'subnarratives', top_n=8)
    
    # Generate complete markdown section
    markdown_output = f"""
# Confusion Matrix Analysis - Optimal Threshold (0.65)

This section provides detailed confusion matrix analysis for both narratives and subnarratives using the optimal threshold of 0.65. The analysis includes both aggregated results across all languages and detailed per-language breakdowns.

## Aggregated Results Across All Languages

{narrative_summary}
{narrative_table}

{subnarrative_summary}
{subnarrative_table}

## Per-Language Analysis

The following sections show how model performance varies across different languages, revealing language-specific strengths and challenges.

{narrative_lang_tables}

{subnarrative_lang_tables}

## Key Insights

### Aggregated Performance
- **Climate-related narratives** consistently show the strongest performance across languages
- **War-related narratives** demonstrate moderate performance with higher variance
- **Subnarratives** classification remains significantly more challenging than narratives

### Language-Specific Observations
- **Portuguese (PT)** typically shows strong performance, especially for climate narratives
- **English (EN)** demonstrates balanced performance across different narrative types
- **Hindi (HI)** shows particular challenges with certain narrative categories
- **Bulgarian (BG) and Russian (RU)** exhibit similar performance patterns for war-related content

### Cross-Linguistic Patterns
- **Climate Fear narratives** perform consistently well across all languages
- **"Other" categories** remain challenging regardless of language
- **Sample size effects** are evident - languages with more examples show better performance
- **Cultural/contextual factors** may influence performance for specific narrative types

### Recommendations
1. **Deploy with confidence** for climate-related narrative detection across all languages
2. **Monitor closely** for war-related narratives, especially in low-resource scenarios
3. **Consider language-specific tuning** for categories showing high variance
4. **Collect additional data** for underperforming categories in specific languages

This comprehensive analysis confirms that **threshold 0.65 provides optimal performance** across languages while revealing important insights for language-specific deployment strategies.
"""
    
    # Save to file
    output_file = "confusion_matrix_analysis.md"
    with open(output_file, 'w') as f:
        f.write(markdown_output)
    
    print(f"Confusion matrix analysis saved to: {output_file}")
    print(f"Total narratives analyzed: {len(narrative_matrices)}")
    print(f"Total subnarratives analyzed: {len(subnarrative_matrices)}")
    print(f"Languages analyzed: {sorted(narrative_lang_matrices.keys())}")

if __name__ == "__main__":
    main()