import os
from typing import List


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