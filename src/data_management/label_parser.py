import json

def parse_json_for_narratives_subnarratives(json_file_path: str):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            hierarchy = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return [], []

    narratives = set()      # To store L1:L2 strings
    subnarratives = set()   # To store L1:L2:L3 strings

    # Iterate through L1 categories (e.g., "URW", "CC")
    for l1_key, l1_value in hierarchy.items():
        if not isinstance(l1_value, dict):
            print(f"Warning: Expected a dictionary for L1 key '{l1_key}', got {type(l1_value)}. Skipping.")
            continue

        # Iterate through L2 categories (e.g., "Blaming the war...")
        for l2_key, l2_value in l1_value.items():
            # Form the narrative string (L1:L2)
            narrative_str = f"{l1_key}: {l2_key}"
            narratives.add(narrative_str)
            narratives.add(f"Other")

            # Check if L2_value is a list (expected for L3 items)
            if isinstance(l2_value, list):
                # Iterate through L3 items (e.g., "Ukraine is the aggressor")
                for l3_item in l2_value:
                    if isinstance(l3_item, str):
                        # Form the sub-narrative string (L1:L2:L3)
                        subnarrative_str = f"{narrative_str}: {l3_item}"
                        subnarratives.add(subnarrative_str)
                    else:
                        print(f"Warning: Expected a string for L3 item under '{narrative_str}', got {type(l3_item)}. Skipping item.")
                # Always add the 'Other' subnarrative for each narrative
                subnarrative_str = f"{narrative_str}: Other"
                subnarratives.add(subnarrative_str)
            else:
                # This L2 category does not have L3 children listed in the expected list format.
                print(f"Info: L2 category '{narrative_str}' does not have a list of L3 children.")

    return sorted(list(narratives)), sorted(list(subnarratives))
    
        
def create_label_mappings(narratives: list, subnarratives: list):
    """
    Creates label-to-ID and ID-to-label mappings for narratives and subnarratives, ensuring unique IDs and preserving hierarchy.

    Args:
        narratives (list): List of narrative strings (L1:L2).
        subnarratives (list): List of subnarrative strings (L1:L2:L3).

    Returns:
        tuple: (label_to_id, id_to_label, narrative_to_subnarrative_ids)
            - label_to_id: dict mapping label string to unique integer ID
            - id_to_label: dict mapping unique integer ID to label string
            - narrative_to_subnarrative_ids: dict mapping narrative string to list of subnarrative IDs
    """
    label_to_id = {}
    id_to_label = {}
    narrative_to_subnarrative_ids = {}
    current_id = 0

    # Assign IDs to narratives
    for narrative in narratives:
        label_to_id[narrative] = current_id
        id_to_label[current_id] = narrative
        current_id += 1

    # Assign IDs to subnarratives and build hierarchy
    for narrative in narratives:
        # Find all subnarratives that start with this narrative
        sub_ids = []
        for subnarrative in subnarratives:
            if subnarrative.startswith(narrative + ":"):
                label_to_id[subnarrative] = current_id
                id_to_label[current_id] = subnarrative
                sub_ids.append(current_id)
                current_id += 1
        narrative_id = label_to_id[narrative]
        narrative_to_subnarrative_ids[narrative_id] = sub_ids

    return label_to_id, id_to_label, narrative_to_subnarrative_ids

def get_label_mappings(json_file_path: str = "data/taxonomy.json"):
    """
    Parses the JSON file and creates label mappings for narratives and subnarratives.

    Args:
        json_file_path (str): Path to the JSON file containing the taxonomy.

    Returns:
        tuple: (label_to_id, id_to_label, narrative_to_subnarrative_ids)
            - label_to_id: dict mapping label string to unique integer ID
            - id_to_label: dict mapping unique integer ID to label string
            - narrative_to_subnarrative_ids: dict mapping narrative string to list of subnarrative IDs
    """
    narratives, subnarratives = parse_json_for_narratives_subnarratives(json_file_path)
    return create_label_mappings(narratives, subnarratives)

if __name__ == "__main__":
    json_file_path = "data/taxonomy.json"  # Replace with your actual JSON file path
    narratives, subnarratives = parse_json_for_narratives_subnarratives(json_file_path)

    print("Narratives:")
    for narrative in narratives:
        print(narrative)

    print("\nSubnarratives:")
    for subnarrative in subnarratives:
        print(subnarrative)

    # Create label mappings with hierarchy
    label_to_id, id_to_label, narrative_to_subnarrative_ids = create_label_mappings(narratives, subnarratives)

    print("\nLabel to ID mapping:")
    print(label_to_id)

    print("\nID to Label mapping:")
    print(id_to_label)

    print("\nNarrative to Subnarrative IDs:")
    for narrative, sub_ids in narrative_to_subnarrative_ids.items():
        print(f"{narrative}: {sub_ids}")