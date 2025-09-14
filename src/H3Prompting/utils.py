import os
from typing import List, Set


def get_processed_files(output_file_path: str) -> Set[str]:
    """
    Parse the output file to get a set of already processed file names.
    
    Args:
        output_file_path: Path to the TSV output file
        
    Returns:
        Set of file names that have already been processed
    """
    processed_files = set()
    try:
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        file_name = line.split('\t')[0]
                        processed_files.add(file_name)
            print(f"[utils] Found {len(processed_files)} already processed files")
        else:
            print(f"[utils] Output file {output_file_path} does not exist, processing all files")
    except Exception as e:
        print(f"[utils] Error reading output file {output_file_path}: {e}")
    
    return processed_files


def get_texts_in_folder(folder_path: str) -> tuple[List[str], List[str]]:
    """
    List all text files in the specified folder.

    Args:
        folder_path: Path to the folder containing text files
    Returns:
        List of texts from the text files
    """
    texts = []
    file_names = []
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
                    file_names.append(file_name)
    except Exception as e:
        print(f"Error reading files from {folder_path}: {e}")
    return texts, file_names

def get_unprocessed_texts(folder_path: str, output_file_path: str) -> tuple[List[str], List[str]]:
    """
    Get texts and file names for files that haven't been processed yet.
    
    Args:
        folder_path: Path to the folder containing text files
        output_file_path: Path to the TSV output file to check for processed files
        
    Returns:
        Tuple of (texts, file_names) for unprocessed files only
    """
    # Get all files and texts
    all_texts, all_file_names = get_texts_in_folder(folder_path)
    
    # Get processed files
    processed_files = get_processed_files(output_file_path)
    
    # Filter out processed files
    unprocessed_texts = []
    unprocessed_file_names = []
    
    for text, file_name in zip(all_texts, all_file_names):
        if file_name not in processed_files:
            unprocessed_texts.append(text)
            unprocessed_file_names.append(file_name)
    
    print(f"[utils] Total files: {len(all_file_names)}, Already processed: {len(processed_files)}, To process: {len(unprocessed_file_names)}")
    
    if unprocessed_file_names:
        print(f"[utils] Files to process: {', '.join(unprocessed_file_names[:5])}{'...' if len(unprocessed_file_names) > 5 else ''}")
    
    return unprocessed_texts, unprocessed_file_names


def get_narratives_for_category(taxonomy: dict, category: str) -> List[str]:
    """
    Retrieve all narratives for a given category from the taxonomy.

    Args:
        taxonomy: The taxonomy dictionary
        category: The category to retrieve narratives for

    Returns:
        A list of narrative names for the specified category
    """
    narratives = []
    if category in taxonomy:
        narratives = list(taxonomy[category].keys())
        
    prefixed_narratives = [f"{category}: {narrative}" for narrative in narratives]
    return prefixed_narratives

def get_subnarratives_for_narrative(taxonomy: dict, category: str, narrative: str) -> List[str]:
    """
    Retrieve all subnarratives for a given narrative from the taxonomy.

    Args:
        taxonomy: The taxonomy dictionary
        category: The category of the narrative
        narrative: The narrative to retrieve subnarratives for
    Returns:
        A list of subnarrative names for the specified narrative
    """
    subnarratives = []
    if category in taxonomy and narrative in taxonomy[category]:
        subnarratives = taxonomy[category][narrative]
        
    prefixed_subnarratives = [f"{category}: {narrative}: {subnarrative}" for subnarrative in subnarratives]
    prefixed_subnarratives.append(f"{category}: {narrative}: Other")
    return prefixed_subnarratives