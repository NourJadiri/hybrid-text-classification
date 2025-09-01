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

# Since this script is in src/scripts, we need to adjust the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def normalize_narratives(narratives):
    """Normalize the narratives field into a clean Python list of strings.

    Handles: list, numpy.ndarray, JSON-encoded list strings, and plain strings
    where items may be separated by common delimiters.
    """
    # numpy arrays -> list
    if isinstance(narratives, np.ndarray):
        narratives = narratives.tolist()

    # JSON string representing a list
    if isinstance(narratives, str):
        try:
            parsed = json.loads(narratives)
            if isinstance(parsed, list):
                return normalize_narratives(parsed)
        except Exception:
            # not JSON; try splitting by common delimiters
            for sep in [';', '|', ',']:
                if sep in narratives:
                    parts = [p.strip() for p in narratives.split(sep) if p.strip()]
                    return parts
            # fallback single-item list
            return [narratives.strip()]

    # list-like -> flatten and clean
    if isinstance(narratives, list):
        cleaned = []
        for item in narratives:
            if item is None:
                continue
            if isinstance(item, (list, np.ndarray)):
                cleaned.extend(normalize_narratives(item))
                continue
            s = str(item).strip()
            if not s:
                continue
            # split items that accidentally contain multiple entries separated by ';' or '|'
            if ';' in s or '|' in s:
                for sep in [';', '|']:
                    if sep in s:
                        cleaned.extend([p.strip() for p in s.split(sep) if p.strip()])
                        break
            else:
                cleaned.append(s)

        # remove duplicates while preserving order
        seen = set()
        result = []
        for x in cleaned:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    # anything else -> empty list
    return []

def analyze_narratives_distribution(df):
    """Analyze the distribution of narratives in the dataset"""
    print("\n=== NARRATIVE DISTRIBUTION ANALYSIS ===")
    
    # Flatten all narratives from all samples (normalize first)
    all_narratives = []
    other_count = 0
    
    for narratives in df['narratives']:
        norm = normalize_narratives(narratives)
        if not norm or norm == ['Other']:
            other_count += 1
        all_narratives.extend(norm)
    
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
        narratives_list = normalize_narratives(row['narratives'])
        narratives_str = "; ".join(narratives_list) if narratives_list else "Other"

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
    
    # Check for narratives formatting (using normalization)
    narrative_issues = 0
    for i, narratives in enumerate(df['narratives']):
        norm = normalize_narratives(narratives)
        if not norm:
            print(f"Row {i}: narratives normalized to empty list from: {type(narratives)} - {narratives}")
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
            narratives_list = normalize_narratives(row['narratives'])
            narratives_str = "; ".join(narratives_list) if narratives_list else "Other"
            
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
    
    # Get the project root (two levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try to find balanced dataset files
    balanced_files = [
        os.path.join(project_root, "data/processed/balanced_sample_400.parquet"),
        os.path.join(project_root, "data/processed/phase0_baseline_labeled.parquet")
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
        
        # Run analyses (only stats)
        analyze_narratives_distribution(df)
        analyze_text_lengths(df)
        check_for_formatting_issues(df)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
