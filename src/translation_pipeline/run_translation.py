"""
Main script to run the translation and consolidation pipeline for subtask-2.
"""
from getpass import getpass
import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Set, cast

from graph import create_translation_graph
from nodes import determine_language_and_translation_need
from annotations import load_all_annotations, write_consolidated_annotations
from state import TranslationState

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


def get_processed_files(output_folder: str) -> Set[str]:
    """
    Get a set of already processed file IDs by checking the output folder.
    
    Args:
        output_folder: Path to the output folder containing translated files
        
    Returns:
        Set of file IDs that have already been processed
    """
    processed_files = set()
    try:
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                if filename.endswith('.txt') and filename != 'subtask-2-annotations.txt':
                    file_id = filename[:-4]  # Remove .txt extension
                    processed_files.add(file_id)
            print(f"[batch] Found {len(processed_files)} already processed files")
        else:
            print(f"[batch] Output folder {output_folder} does not exist, processing all files")
    except Exception as e:
        print(f"[batch] Error checking processed files in {output_folder}: {e}")
    
    return processed_files

def get_all_text_files(data_folder: str) -> List[Tuple[str, str, str]]:
    """
    Get all text files from all language folders.
    
    Args:
        data_folder: Path to main data folder
        
    Returns:
        List of tuples (file_path, file_id, source_language)
    """
    all_files = []
    language_folders = ["EN", "BG", "HI", "PT", "RU"]
    
    for lang in language_folders:
        lang_folder = os.path.join(data_folder, lang, "raw-documents")
        if not os.path.exists(lang_folder):
            print(f"[batch] Warning: Language folder not found: {lang_folder}")
            continue
            
        print(f"[batch] Scanning {lang_folder}...")
        
        for filename in os.listdir(lang_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(lang_folder, filename)
                file_id = filename[:-4]  # Remove .txt extension
                all_files.append((file_path, file_id, lang))
    
    print(f"[batch] Found {len(all_files)} text files total")
    return all_files


def read_text_file(file_path: str) -> str:
    """Read text content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"[batch] Error reading {file_path}: {e}")
        return ""


def prepare_initial_states(text_files: List[Tuple[str, str, str]], processed_files: Set[str]) -> List[dict]:
    """
    Prepare initial states for batch processing, skipping already processed files.
    
    Args:
        text_files: List of (file_path, file_id, source_language) tuples
        processed_files: Set of file IDs that have already been processed
        
    Returns:
        List of initial states for the graph
    """
    initial_states = []
    skipped_count = 0
    
    for file_path, file_id, source_language in text_files:
        # Skip if already processed
        if file_id in processed_files:
            skipped_count += 1
            continue
            
        text_content = read_text_file(file_path)
        if not text_content:
            print(f"[batch] Skipping empty file: {file_path}")
            continue
            
        initial_state = {
            "text": text_content,
            "file_id": file_id,
            "source_language": source_language
        }
        initial_states.append(initial_state)
    
    print(f"[batch] Prepared {len(initial_states)} initial states for processing")
    if skipped_count > 0:
        print(f"[batch] Skipped {skipped_count} already processed files")
    return initial_states


async def main():
    """Main function to run the translation pipeline."""
    # Configuration
    DATA_FOLDER = "/home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification/data"
    OUTPUT_FOLDER = "/home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification/data/subtask-2-translated"
    OUTPUT_ANNOTATIONS = "/home/twoface/Documents/Passau/masterarbeit/hybrid-text-classification/data/subtask-2-translated/subtask-2-annotations.txt"
    
    print("[batch] Starting translation pipeline for subtask-2...")
    
    # Step 1: Load all annotations first
    print("[batch] Loading annotations from all language folders...")
    all_annotations = load_all_annotations(DATA_FOLDER)
    
    # Step 2: Write consolidated annotations
    print("[batch] Writing consolidated annotations...")
    write_consolidated_annotations(all_annotations, OUTPUT_ANNOTATIONS)
    
    # Step 3: Get all text files
    print("[batch] Collecting all text files...")
    text_files = get_all_text_files(DATA_FOLDER)
    
    # Step 4: Filter files that have annotations (for subtask-2)
    files_with_annotations = []
    for file_path, file_id, source_language in text_files:
        if file_id in all_annotations:
            files_with_annotations.append((file_path, file_id, source_language))
        else:
            print(f"[batch] Skipping {file_id} - no subtask-2 annotation found")
    
    print(f"[batch] Processing {len(files_with_annotations)} files with subtask-2 annotations")
    
    # Step 5: Check for already processed files
    processed_files = get_processed_files(OUTPUT_FOLDER)
    
    # Step 6: Prepare initial states for batch processing (excluding already processed)
    initial_states = prepare_initial_states(files_with_annotations, processed_files)
    
    if not initial_states:
        print("[batch] No new files to process. All files already processed or no files found.")
        return
    
    # Step 7: Create the translation graph
    translation_graph = create_translation_graph(OUTPUT_FOLDER)
    
    # Step 8: Process all files in batch
    print(f"[batch] Starting batch processing of {len(initial_states)} files...")
    
    # Cast for type safety
    typed_states = cast(List[TranslationState], initial_states)
    
    try:
        results = await translation_graph.abatch(typed_states)  # type: ignore[arg-type]
        print(f"[batch] Batch processing completed. Processed {len(results)} files.")
        
        # Count successful vs failed
        successful = sum(1 for result in results if not result.get("error_message"))
        failed = len(results) - successful
        
        print(f"[batch] Results: {successful} successful, {failed} failed")
        
    except Exception as e:
        print(f"[batch] Error during batch processing: {e}")
        raise
    
    print("[batch] Translation pipeline completed!")
    print(f"[batch] Translated texts saved in: {OUTPUT_FOLDER}")
    print(f"[batch] Consolidated annotations saved in: {OUTPUT_ANNOTATIONS}")


if __name__ == "__main__":
    asyncio.run(main())