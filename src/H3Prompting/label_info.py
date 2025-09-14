import pandas as pd
import json
from typing import Dict, List

def load_taxonomy(file_path: str = "data/taxonomy.json") -> Dict[str, Dict[str, List[str]]]:
    """
    Load taxonomy from a JSON file.

    Args:
        file_path: Path to the taxonomy JSON file
    Returns:
        Dictionary mapping categories to their themes and narratives
    """
    try:
        with open(file_path, 'r') as f:
            taxonomy = json.load(f)
        return taxonomy
    except FileNotFoundError:
        print(f"Warning: Taxonomy file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return {}

def flatten_taxonomy(taxonomy: dict) -> tuple[set[str], set[str]]:
    flat_narratives = []
    flat_subnarratives = []

    for category, narratives in taxonomy.items():
        for narrative, subnarratives in narratives.items():
            flat_narrative_str = f"{category}: {narrative}"
            flat_narratives.append(flat_narrative_str)

            for subnarrative in subnarratives:
                flat_subnarrative_str = f"{flat_narrative_str}: {subnarrative}"
                flat_subnarratives.append(flat_subnarrative_str)

            other_subnarrative_str = f"{flat_narrative_str}: Other"
            flat_subnarratives.append(other_subnarrative_str)

    return set(flat_narratives), set(flat_subnarratives)

def load_narrative_definitions(file_path: str = "data/narrative_definitions.csv") -> Dict[str, Dict[str, str]]:
    """
    Load narrative definitions from CSV file.

    Args:
        file_path: Path to the narrative definitions CSV file

    Returns:
        Dictionary mapping narrative names to their definitions and examples
    """
    try:
        df = pd.read_csv(file_path)
        definitions = {}

        for _, row in df.iterrows():
            narrative = row.get('narrative') if pd.notna(row.get('narrative', None)) else None
            if narrative is None:
                continue

            definition = row.get('definition', '') if pd.notna(row.get('definition', '')) else ''
            example = row.get('example', '') if pd.notna(row.get('example', '')) else ''
            instruction = row.get('instruction for annotator', '') if pd.notna(row.get('instruction for annotator', '')) else ''

            definitions[narrative] = {
                'definition': definition,
                'example': example,
                'instruction': instruction
            }

        return definitions

    except FileNotFoundError:
        print(f"Warning: Narrative definitions file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading narrative definitions: {e}")
        return {}


def get_unique_narratives_from_definitions(definitions_path: str = "data/narrative_definitions.csv") -> List[str]:
    """
    Extract unique narrative names from the definitions file.

    Args:
        definitions_path: Path to the narrative definitions CSV file

    Returns:
        List of unique narrative names
    """
    try:
        df = pd.read_csv(definitions_path)
        # Remove duplicates while preserving order
        narratives = df['narrative'].drop_duplicates().tolist()
        return narratives
    except Exception as e:
        print(f"Error extracting narratives: {e}")
        return []


def load_subnarrative_definitions(file_path: str = "data/subnarrative_definitions.csv") -> Dict[str, Dict[str, str]]:
    """
    Load subnarrative definitions from CSV file.

    Args:
        file_path: Path to the subnarrative definitions CSV file

    Returns:
        Dictionary mapping subnarrative names to their definitions and examples
    """
    try:
        df = pd.read_csv(file_path)
        definitions = {}

        for _, row in df.iterrows():
            subnarrative = row.get('subnarrative') if pd.notna(row.get('subnarrative', None)) else None
            if subnarrative is None:
                continue

            definition = row.get('definition', '') if pd.notna(row.get('definition', '')) else ''
            example = row.get('example', '') if pd.notna(row.get('example', '')) else ''
            instruction = row.get('instruction for annotator', '') if pd.notna(row.get('instruction for annotator', '')) else ''

            definitions[subnarrative] = {
                'definition': definition,
                'example': example,
                'instruction': instruction
            }

        return definitions

    except FileNotFoundError:
        print(f"Warning: Subnarrative definitions file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading subnarrative definitions: {e}")
        return {}


def get_unique_subnarratives_from_definitions(definitions_path: str = "data/subnarrative_definitions.csv") -> List[str]:
    """
    Extract unique subnarrative names from the definitions file.

    Args:
        definitions_path: Path to the subnarrative definitions CSV file

    Returns:
        List of unique subnarrative names
    """
    try:
        df = pd.read_csv(definitions_path)
        subnarratives = df['subnarrative'].drop_duplicates().tolist()
        return subnarratives
    except Exception as e:
        print(f"Error extracting subnarratives: {e}")
        return []

def print_sample_definitions(n: int = 5,
                             narr_def_path: str = "data/narrative_definitions.csv",
                             subnarr_def_path: str = "data/subnarrative_definitions.csv") -> None:
    """Print a small sample of narratives and subnarratives with their definitions and examples.

    This is a non-runnable helper (no CLI). Call it from a REPL or another script.

    Args:
        n: number of samples to show for each category
        narr_def_path: path to narrative definitions CSV
        subnarr_def_path: path to subnarrative definitions CSV
    """
    narr_defs = load_narrative_definitions(narr_def_path)
    sub_defs = load_subnarrative_definitions(subnarr_def_path)

    def _print_samples(defs: Dict[str, Dict[str, str]], title: str):
        keys = list(defs.keys())
        print(f"\n{title} (showing up to {n} samples, total={len(keys)})")
        for i, k in enumerate(keys[:n], 1):
            v = defs.get(k, {})
            definition = v.get('definition', '')
            example = v.get('example', '')
            print(f"\n{i}. {k}")
            if definition:
                print(f"   Definition: {definition}")
            if example:
                print(f"   Example: {example}")

    _print_samples(narr_defs, "Narratives")
    _print_samples(sub_defs, "Subnarratives")

    print("\nDone.")