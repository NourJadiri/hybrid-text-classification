#!/usr/bin/env python3
"""
Script to analyze the balanced dataset and understand class distribution
to debug why the Qwen model is predicting "other" for everything.
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import sys
import os

# Add src to path to import utilities
sys.path.append('src')

def analyze_narratives_distribution(df):
    """Analyze the distribution of narratives in the dataset"""
    print("\n=== NARRATIVE DISTRIBUTION ANALYSIS ===")
    
    # Flatten all narratives from all samples
    all_narratives = []
    other_count = 0
    
    for narratives in df['narratives']:
        if isinstance(narratives, list):
            if not narratives or narratives == ['Other']:
                other_count += 1
            all_narratives.extend(narratives)
        else:
            print(f"Unexpected narrative format: {narratives}")
    
    # Count occurrences
    narrative_counts = Counter(all_narratives)
    
    print(f"Total samples: {len(df)}")
    print(f"Samples with 'Other' or empty narratives: {other_count} ({other_count/len(df)*100:.1f}%)")
    print(f"Total narrative occurrences: {len(all_narratives)}")
    print(f"Unique narratives: {len(narrative_counts)}")
    
    print("\nTop 20 most common narratives:")
    for narrative, count in narrative_counts.most_common(20):
        percentage = count / len(all_narratives) * 100
        print(f"  {narrative}: {count} ({percentage:.1f}%)")
    
    return narrative_counts

def analyze_sample_examples(df, n_samples=10):
    """Show some sample texts and their narratives"""
    print(f"\n=== SAMPLE EXAMPLES (first {n_samples}) ===")
    
    for i in range(min(n_samples, len(df))):
        row = df.iloc[i]
        narratives_str = "; ".join(row['narratives']) if isinstance(row['narratives'], list) else str(row['narratives'])
        
        print(f"\nSample {i+1}:")
        print(f"Text: {row['text'][:200]}{'...' if len(row['text']) > 200 else ''}")
        print(f"Narratives: [{narratives_str}]")

def analyze_text_lengths(df):
    """Analyze text length distribution"""
    print("\n=== TEXT LENGTH ANALYSIS ===")
    
    text_lengths = df['text'].str.len()
    
    print(f"Average text length: {text_lengths.mean():.1f} characters")
    print(f"Median text length: {text_lengths.median():.1f} characters")
    print(f"Min text length: {text_lengths.min()} characters")
    print(f"Max text length: {text_lengths.max()} characters")
    print(f"Texts > 1000 chars: {(text_lengths > 1000).sum()} ({(text_lengths > 1000).sum()/len(df)*100:.1f}%)")

def check_for_formatting_issues(df):
    """Check for potential formatting issues that might confuse the model"""
    print("\n=== FORMATTING ISSUES CHECK ===")
    
    # Check for empty or very short texts
    empty_texts = df[df['text'].str.len() < 10]
    print(f"Texts shorter than 10 characters: {len(empty_texts)}")
    
    # Check for narratives formatting
    narrative_issues = 0
    for i, narratives in enumerate(df['narratives']):
        if not isinstance(narratives, list):
            print(f"Row {i}: narratives is not a list: {type(narratives)} - {narratives}")
            narrative_issues += 1
        elif len(narratives) == 0:
            print(f"Row {i}: empty narratives list")
            narrative_issues += 1
    
    print(f"Total narrative formatting issues: {narrative_issues}")

def analyze_training_format_preview(df, n_samples=3):
    """Show how the data would look in the training format"""
    print(f"\n=== TRAINING FORMAT PREVIEW ===")
    
    # Import the prompt template
    try:
        from utils.prompt_template import create_comprehensive_prompt_template
        prompt_template = create_comprehensive_prompt_template()
        
        for i in range(min(n_samples, len(df))):
            row = df.iloc[i]
            narratives_str = "; ".join(row['narratives']) if isinstance(row['narratives'], list) and row['narratives'] else "Other"
            
            user_content = prompt_template.format(text=row['text'])
            
            print(f"\n--- Training Example {i+1} ---")
            print("USER MESSAGE:")
            print(user_content)
            print("\nASSISTANT MESSAGE:")
            print(f"[{narratives_str}]")
            print("-" * 50)
    
    except ImportError as e:
        print(f"Could not import prompt template: {e}")

def main():
    print("=== BALANCED DATASET ANALYSIS ===")
    
    # Try to find balanced dataset files
    balanced_files = [
        "data/processed/balanced_sample_400.parquet",
        "data/processed/phase0_baseline_labeled.parquet"
    ]
    
    dataset_path = None
    for file_path in balanced_files:
        if os.path.exists(file_path):
            dataset_path = file_path
            break
    
    if not dataset_path:
        print("No balanced dataset found. Available files:")
        for file_path in balanced_files:
            print(f"  {file_path}: {'EXISTS' if os.path.exists(file_path) else 'NOT FOUND'}")
        return
    
    print(f"Loading dataset: {dataset_path}")
    
    try:
        df = pd.read_parquet(dataset_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if this looks like a balanced sample (only text and narratives)
        if list(df.columns) == ['text', 'narratives']:
            print("✓ This appears to be a balanced sample dataset")
        else:
            print("⚠ This appears to be a full dataset, not a balanced sample")
            # Filter to only text and narratives if it's a full dataset
            if 'text' in df.columns and 'narratives' in df.columns:
                df = df[['text', 'narratives']].dropna().reset_index(drop=True)
                print(f"Filtered to text and narratives columns. New shape: {df.shape}")
        
        # Run analyses
        analyze_narratives_distribution(df)
        analyze_text_lengths(df)
        check_for_formatting_issues(df)
        analyze_sample_examples(df, n_samples=5)
        analyze_training_format_preview(df, n_samples=2)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
