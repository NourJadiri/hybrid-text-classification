#!/usr/bin/env python3
"""
Simple script to add annotations for translated files to unified-annotations.tsv
"""

import os
from pathlib import Path

def main():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    texts_dir = project_root / "data/all-texts-unified/texts"
    unified_file = project_root / "data/all-texts-unified/unified-annotations.tsv"
    
    # Find translated files
    translated_files = []
    for file in texts_dir.glob("*_TRANS_*.txt"):
        translated_files.append(file.stem)
    
    if not translated_files:
        print("No translated files found")
        return
        
    print(f"Found {len(translated_files)} translated files")
    
    # Load existing annotations to avoid duplicates
    existing = set()
    if unified_file.exists():
        with open(unified_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing.add(line.split('\t')[0])
    
    # Process each translated file
    new_annotations = []
    for file_id in translated_files:
        if file_id in existing:
            print(f"Skipping existing: {file_id}")
            continue
            
        # Parse file ID: SOURCE_TRANS_TARGET_ORIGINAL_FILE
        parts = file_id.split('_', 3)
        if len(parts) < 4:
            print(f"Skipping malformed: {file_id}")
            continue
            
        source_lang = parts[0]
        original_file = parts[3] + '.txt'  # Add .txt extension
        print(f"Processing: {file_id} -> {original_file}")
        
        # Load original annotation
        source_annotations = project_root / f"data/{source_lang}/subtask-2-annotations.txt"
        if not source_annotations.exists():
            print(f"No source annotations for: {source_lang}")
            continue
            
        # Find annotation for original file
        found = False
        with open(source_annotations, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(original_file + '\t'):
                    parts = line.split('\t')
                    if len(parts) == 3:
                        narratives = parts[1]
                        subnarratives = parts[2]
                        new_annotations.append(f"{file_id}\t{narratives}\t{subnarratives}")
                        print(f"Added: {file_id}")
                        found = True
                    break
        if not found:
            print(f"No annotation found for: {original_file}")
    
    # Append new annotations
    if new_annotations:
        with open(unified_file, 'a', encoding='utf-8') as f:
            for annotation in new_annotations:
                f.write(annotation + '\n')
        print(f"Added {len(new_annotations)} new annotations")
    else:
        print("No new annotations to add")

if __name__ == "__main__":
    main()