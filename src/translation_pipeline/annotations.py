"""
Annotation processing utilities for consolidating subtask-2 annotations from all language folders.
"""
import os
from typing import Dict, List, Set
from pathlib import Path


def load_annotations_from_folder(language_folder: str) -> Dict[str, str]:
    """
    Load subtask-2 annotations from a specific language folder.
    
    Args:
        language_folder: Path to language folder (e.g., "/path/to/data/EN")
    
    Returns:
        Dictionary mapping file_id (without .txt extension) to annotation line
    """
    annotations_file = os.path.join(language_folder, "subtask-2-annotations.txt")
    annotations = {}
    
    if not os.path.exists(annotations_file):
        print(f"[annotations] Warning: No subtask-2-annotations.txt found in {language_folder}")
        return annotations
    
    print(f"[annotations] Loading annotations from {annotations_file}")
    
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse the annotation line: file_id\tnarrative\tsubnarrative
                parts = line.split('\t')
                if len(parts) >= 3:
                    file_id = parts[0]
                    # Remove .txt extension if present for consistent key format
                    if file_id.endswith('.txt'):
                        file_id = file_id[:-4]
                    
                    annotations[file_id] = line
                else:
                    print(f"[annotations] Warning: Malformed line {line_num} in {annotations_file}: {line}")
    
    except Exception as e:
        print(f"[annotations] Error reading {annotations_file}: {e}")
    
    print(f"[annotations] Loaded {len(annotations)} annotations from {language_folder}")
    return annotations


def load_all_annotations(data_folder: str) -> Dict[str, str]:
    """
    Load subtask-2 annotations from all language folders.
    
    Args:
        data_folder: Path to main data folder containing language subfolders
    
    Returns:
        Dictionary mapping file_id to annotation line
    """
    all_annotations = {}
    language_folders = ["EN", "BG", "HI", "PT", "RU"]
    
    for lang in language_folders:
        lang_folder = os.path.join(data_folder, lang)
        if os.path.exists(lang_folder):
            lang_annotations = load_annotations_from_folder(lang_folder)
            
            # Check for duplicate file IDs across languages
            duplicates = set(all_annotations.keys()) & set(lang_annotations.keys())
            if duplicates:
                print(f"[annotations] Warning: Duplicate file IDs found between languages: {duplicates}")
            
            all_annotations.update(lang_annotations)
        else:
            print(f"[annotations] Warning: Language folder not found: {lang_folder}")
    
    print(f"[annotations] Total annotations loaded: {len(all_annotations)}")
    return all_annotations


def write_consolidated_annotations(annotations: Dict[str, str], output_file: str) -> None:
    """
    Write consolidated annotations to a single file.
    
    Args:
        annotations: Dictionary of file_id -> annotation_line
        output_file: Path to output annotation file
    """
    print(f"[annotations] Writing consolidated annotations to {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Sort by file ID for consistent ordering
            for file_id in sorted(annotations.keys()):
                annotation_line = annotations[file_id]
                f.write(annotation_line + '\n')
        
        print(f"[annotations] Successfully wrote {len(annotations)} annotations to {output_file}")
    
    except Exception as e:
        print(f"[annotations] Error writing consolidated annotations: {e}")
        raise


def get_files_with_annotations(annotations: Dict[str, str]) -> Set[str]:
    """
    Get set of file IDs that have annotations.
    
    Args:
        annotations: Dictionary of file_id -> annotation_line
    
    Returns:
        Set of file IDs that have annotations
    """
    return set(annotations.keys())


def extract_file_id_from_path(file_path: str) -> str:
    """
    Extract file ID from a file path.
    
    Args:
        file_path: Full path to text file
    
    Returns:
        File ID without extension (e.g., "EN_CC_100013")
    """
    filename = os.path.basename(file_path)
    if filename.endswith('.txt'):
        return filename[:-4]
    return filename


def validate_annotations_coverage(text_files: List[str], annotations: Dict[str, str]) -> tuple[List[str], List[str]]:
    """
    Validate that we have annotations for the text files we're processing.
    
    Args:
        text_files: List of text file paths
        annotations: Dictionary of file_id -> annotation_line
    
    Returns:
        Tuple of (files_with_annotations, files_without_annotations)
    """
    files_with_annotations = []
    files_without_annotations = []
    
    for file_path in text_files:
        file_id = extract_file_id_from_path(file_path)
        
        if file_id in annotations:
            files_with_annotations.append(file_path)
        else:
            files_without_annotations.append(file_path)
    
    print(f"[annotations] Validation results:")
    print(f"  Files with annotations: {len(files_with_annotations)}")
    print(f"  Files without annotations: {len(files_without_annotations)}")
    
    if files_without_annotations:
        print(f"[annotations] Files without annotations: {files_without_annotations[:10]}...")  # Show first 10
    
    return files_with_annotations, files_without_annotations