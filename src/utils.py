import os
from typing import List


def get_texts_in_folder(folder_path: str) -> List[str]:
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