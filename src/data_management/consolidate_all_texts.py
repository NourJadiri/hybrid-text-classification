#!/usr/bin/env python3
"""
Script to consolidate all original and translated texts into a unified dataset.

This script:
1. Copies all original texts from language folders with clear naming
2. Copies all translated texts from the translation pipeline 
3. Creates a unified annotation file mapping all texts to their narratives
4. Includes metadata about language and translation status
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd


def load_annotations(file_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Load annotations from a TSV file.
    Returns: {filename: (narratives, subnarratives)}
    """
    annotations = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: Annotation file not found: {file_path}")
        return annotations
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                filename = parts[0]
                narratives = parts[1]
                subnarratives = parts[2]
                annotations[filename] = (narratives, subnarratives)
    
    return annotations


def get_all_original_files(data_folder: str) -> List[Tuple[str, str, str]]:
    """
    Get all original text files from language folders.
    Returns: [(language, source_path, filename)]
    """
    files = []
    languages = ['EN', 'BG', 'HI', 'PT', 'RU']
    
    for lang in languages:
        lang_folder = os.path.join(data_folder, lang, 'raw-documents')
        if os.path.exists(lang_folder):
            for filename in os.listdir(lang_folder):
                if filename.endswith('.txt'):
                    source_path = os.path.join(lang_folder, filename)
                    files.append((lang, source_path, filename))
    
    print(f"Found {len(files)} original text files across all languages")
    return files


def get_all_translated_files(translated_folder: str) -> List[Tuple[str, str]]:
    """
    Get all translated text files.
    Returns: [(source_path, filename)]
    """
    files = []
    
    if os.path.exists(translated_folder):
        for filename in os.listdir(translated_folder):
            if filename.endswith('.txt') and filename != 'subtask-2-annotations.txt':
                source_path = os.path.join(translated_folder, filename)
                files.append((source_path, filename))
    
    print(f"Found {len(files)} translated text files")
    return files


def load_all_annotations(data_folder: str) -> Dict[str, Tuple[str, str]]:
    """
    Load annotations from all language folders.
    Returns: {filename: (narratives, subnarratives)}
    """
    all_annotations = {}
    languages = ['EN', 'BG', 'HI', 'PT', 'RU']
    
    for lang in languages:
        annotation_file = os.path.join(data_folder, lang, 'subtask-2-annotations.txt')
        lang_annotations = load_annotations(annotation_file)
        print(f"Loaded {len(lang_annotations)} annotations for {lang}")
        all_annotations.update(lang_annotations)
    
    # Also load translated annotations
    translated_annotations = load_annotations(os.path.join(data_folder, 'subtask-2-translated', 'subtask-2-annotations.txt'))
    print(f"Loaded {len(translated_annotations)} translated annotations")
    all_annotations.update(translated_annotations)
    
    print(f"Total unique annotations: {len(all_annotations)}")
    return all_annotations


def create_unified_dataset(data_folder: str, output_folder: str):
    """
    Create a unified dataset with all original and translated texts.
    """
    print(f"Creating unified dataset in: {output_folder}")
    
    # Create output directories
    texts_folder = os.path.join(output_folder, 'texts')
    os.makedirs(texts_folder, exist_ok=True)
    
    # Load all annotations
    all_annotations = load_all_annotations(data_folder)
    
    # Prepare unified annotation data
    unified_annotations = []
    copied_files = set()
    
    # 1. Copy original texts with language prefixes
    print("\n=== Copying original texts ===")
    original_files = get_all_original_files(data_folder)
    
    for lang, source_path, filename in original_files:
        # Create new filename with language prefix
        new_filename = f"{lang}_ORIG_{filename}"
        dest_path = os.path.join(texts_folder, new_filename)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        copied_files.add(new_filename)
        
        # Get annotations for this file
        if filename in all_annotations:
            narratives, subnarratives = all_annotations[filename]
            unified_annotations.append({
                'filename': new_filename,
                'original_filename': filename,
                'language': lang,
                'type': 'original',
                'narratives': narratives,
                'subnarratives': subnarratives
            })
        else:
            print(f"Warning: No annotations found for {filename}")
    
    print(f"Copied {len(original_files)} original files")
    
    # 2. Copy translated texts
    print("\n=== Copying translated texts ===")
    translated_folder = os.path.join(data_folder, 'subtask-2-translated')
    translated_files = get_all_translated_files(translated_folder)
    
    for source_path, filename in translated_files:
        # For translated files, keep original filename but add TRANS suffix if it conflicts
        new_filename = filename
        if new_filename in copied_files:
            # This shouldn't happen, but handle conflicts
            new_filename = f"TRANS_{filename}"
        
        dest_path = os.path.join(texts_folder, new_filename)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        copied_files.add(new_filename)
        
        # Determine original language from filename patterns
        original_lang = determine_original_language(filename)
        
        # Get annotations for this file
        if filename in all_annotations:
            narratives, subnarratives = all_annotations[filename]
            unified_annotations.append({
                'filename': new_filename,
                'original_filename': filename,
                'language': original_lang,
                'type': 'translated' if original_lang != 'EN' else 'original',
                'narratives': narratives,
                'subnarratives': subnarratives
            })
        else:
            print(f"Warning: No annotations found for {filename}")
    
    print(f"Copied {len(translated_files)} translated files")
    
    # 3. Create unified annotation files
    print("\n=== Creating unified annotations ===")
    
    # Create DataFrame for easy manipulation
    df = pd.DataFrame(unified_annotations)
    
    # Save as TSV (traditional format)
    tsv_path = os.path.join(output_folder, 'unified-annotations.tsv')
    with open(tsv_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['filename']}\t{row['narratives']}\t{row['subnarratives']}\n")
    
    # Save as CSV with metadata (more detailed)
    csv_path = os.path.join(output_folder, 'unified-annotations-detailed.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 4. Create summary statistics
    print("\n=== Creating summary ===")
    summary = {
        'total_files': len(df),
        'by_language': df['language'].value_counts().to_dict(),
        'by_type': df['type'].value_counts().to_dict(),
        'unique_narratives': df['narratives'].nunique(),
        'unique_subnarratives': df['subnarratives'].nunique()
    }
    
    summary_path = os.path.join(output_folder, 'dataset-summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("UNIFIED DATASET SUMMARY\n")
        f.write("=====================\n\n")
        f.write(f"Total files: {summary['total_files']}\n\n")
        
        f.write("By Language:\n")
        for lang, count in summary['by_language'].items():
            f.write(f"  {lang}: {count}\n")
        
        f.write("\nBy Type:\n")
        for type_name, count in summary['by_type'].items():
            f.write(f"  {type_name}: {count}\n")
        
        f.write(f"\nUnique narrative categories: {summary['unique_narratives']}\n")
        f.write(f"Unique subnarrative categories: {summary['unique_subnarratives']}\n")
    
    print(f"\n=== CONSOLIDATION COMPLETE ===")
    print(f"Output folder: {output_folder}")
    print(f"Text files: {texts_folder}")
    print(f"Annotations (TSV): {tsv_path}")
    print(f"Annotations (CSV): {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"\nTotal files consolidated: {len(df)}")
    print(f"Languages: {list(summary['by_language'].keys())}")
    print(f"Types: {list(summary['by_type'].keys())}")


def determine_original_language(filename: str) -> str:
    """
    Determine the original language of a file based on filename patterns.
    """
    if filename.startswith('EN_'):
        return 'EN'
    elif filename.startswith('BG_') or 'BG_' in filename:
        return 'BG'
    elif filename.startswith('HI_'):
        return 'HI'
    elif filename.startswith('PT_'):
        return 'PT'
    elif filename.startswith('RU-') or filename.startswith('RU_'):
        return 'RU'
    elif 'CC_BG_' in filename or 'URW_BG_' in filename:
        return 'BG'
    else:
        # Default to EN for files that don't have clear language indicators
        return 'EN'


def main():
    """Main function."""
    # Configuration
    DATA_FOLDER = "/home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification/data"
    OUTPUT_FOLDER = "/home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification/data/all-texts-unified"
    
    print("UNIFIED DATASET CONSOLIDATION")
    print("============================")
    print(f"Source folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Create unified dataset
    create_unified_dataset(DATA_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()