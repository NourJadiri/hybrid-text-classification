import re

def extract_category(text: str) -> str:
    """
    Extract the category from the given text using a prompt template.
    
    Args:
        text: The input text to classify
    """
    match = re.search(r'\[([^\[\]]+)\]', text)
    return match.group(1).strip() if match else ""

def extract_narratives(text: str) -> list[str]:
    """
    Extract narratives from the given text using a prompt template.
    
    Args:
        text: The input text to classify
    """
    match = re.search(r'\[([^\[\]]+)\]', text)
    if match:
        narratives = match.group(1).split(';')
        return [narrative.strip() for narrative in narratives]
    return []

def extract_subnarratives(text: str) -> list[str]:
    """
    Extract subnarratives from the given text using a prompt template.
    
    Args:
        text: The input text to classify
    """
    match = re.search(r'\[([^\[\]]+)\]', text)
    if match:
        subnarratives = match.group(1).split(';')
        return [subnarrative.strip() for subnarrative in subnarratives]
    return []