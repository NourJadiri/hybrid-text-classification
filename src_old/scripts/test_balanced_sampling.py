#!/usr/bin/env python3
# src/scripts/test_balanced_sampling.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.balanced_sampling import (
    create_balanced_sample_for_finetuning,
    analyze_class_distribution,
    create_balanced_multilabel_sample
)
from data_management.loaders import load_labeled_df

def main():
    parser = argparse.ArgumentParser(description="Test and demonstrate balanced sampling functionality.")
    
    parser.add_argument("--dataset_path", type=str, default="data/processed/phase0_baseline_labeled.parquet",
                        help="Path to the labeled parquet dataset.")
    parser.add_argument("--target_size", type=int, default=500,
                        help="Target size for the balanced sample.")
    parser.add_argument("--max_other_percentage", type=float, default=0.15,
                        help="Maximum percentage of 'Other' class (0.0-1.0).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the balanced sample (optional).")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze the dataset distribution without creating a sample.")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BALANCED SAMPLING TEST")
    print("="*60)
    
    # Load the dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        df = load_labeled_df(args.dataset_path)
        print(f"Successfully loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Analyze the original distribution
    print(f"\n{'-'*40}")
    print("ORIGINAL DATASET ANALYSIS")
    print(f"{'-'*40}")
    
    stats = analyze_class_distribution(df, 'narratives')
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Unique classes: {stats['unique_classes']}")
    print(f"'Other' class count: {stats['other_count']:,}")
    print(f"'Other' class percentage: {stats['other_percentage']:.1f}%")
    
    print(f"\nTop 10 most common classes:")
    for i, (class_name, count) in enumerate(stats['most_common'][:10], 1):
        percentage = (count / sum(stats['class_counts'].values())) * 100
        print(f"  {i:2d}. {class_name:30s} {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nTop 10 least common classes:")
    for i, (class_name, count) in enumerate(stats['most_common'][-10:], 1):
        percentage = (count / sum(stats['class_counts'].values())) * 100
        print(f"  {i:2d}. {class_name:30s} {count:6,} ({percentage:5.1f}%)")
    
    if args.analyze_only:
        print("\nAnalysis complete. Use --analyze_only=False to create a balanced sample.")
        return
    
    # Create balanced sample
    print(f"\n{'-'*40}")
    print("CREATING BALANCED SAMPLE")
    print(f"{'-'*40}")
    print(f"Target size: {args.target_size:,}")
    print(f"Max 'Other' percentage: {args.max_other_percentage*100:.1f}%")
    
    try:
        balanced_sample = create_balanced_multilabel_sample(
            df=df,
            label_column='narratives',
            text_column='text',
            target_size=args.target_size,
            min_samples_per_class=5,
            max_other_percentage=args.max_other_percentage,
            prioritize_rare_classes=True,
            random_state=42
        )
        
        print(f"\nSuccessfully created balanced sample with {len(balanced_sample)} samples")
        
        # Analyze the balanced sample
        print(f"\n{'-'*40}")
        print("BALANCED SAMPLE ANALYSIS")
        print(f"{'-'*40}")
        
        balanced_stats = analyze_class_distribution(balanced_sample, 'narratives')
        print(f"Sample size: {len(balanced_sample):,}")
        print(f"Unique classes: {balanced_stats['unique_classes']}")
        print(f"'Other' class count: {balanced_stats['other_count']:,}")
        print(f"'Other' class percentage: {balanced_stats['other_percentage']:.1f}%")
        
        print(f"\nClass distribution in balanced sample:")
        for i, (class_name, count) in enumerate(balanced_stats['most_common'][:15], 1):
            percentage = (count / sum(balanced_stats['class_counts'].values())) * 100
            print(f"  {i:2d}. {class_name:30s} {count:6,} ({percentage:5.1f}%)")
        
        # Compare improvements
        print(f"\n{'-'*40}")
        print("IMPROVEMENT SUMMARY")
        print(f"{'-'*40}")
        
        original_other_pct = stats['other_percentage']
        balanced_other_pct = balanced_stats['other_percentage']
        reduction = original_other_pct - balanced_other_pct
        
        print(f"'Other' class reduction: {original_other_pct:.1f}% → {balanced_other_pct:.1f}% ({reduction:+.1f}%)")
        print(f"Classes represented: {balanced_stats['unique_classes']} out of {stats['unique_classes']}")
        
        # Save if requested
        if args.output_path:
            balanced_sample.to_parquet(args.output_path, index=False)
            print(f"\nBalanced sample saved to: {args.output_path}")
            
    except Exception as e:
        print(f"Error creating balanced sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
