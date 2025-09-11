def get_label_indices_by_level(id_to_label_map: dict) -> dict:
    """
    Categorizes label indices into 'narrative' and 'subnarrative' levels.

    Args:
        id_to_label_map (dict): The dictionary mapping integer IDs to label strings.

    Returns:
        dict: A dictionary containing two keys:
              'narrative_indices' (list): A sorted list of indices for L1:L2 labels.
              'subnarrative_indices' (list): A sorted list of indices for L1:L2:L3 labels.
    """
    print("Categorizing label indices by hierarchy level...")

    narrative_indices = []
    subnarrative_indices = []

    # Ensure we iterate through all possible indices from 0 to max_id
    if not id_to_label_map:
        return {'narrative_indices': [], 'subnarrative_indices': []}
        
    num_labels = max(id_to_label_map.keys()) + 1

    for idx in range(num_labels):
        label_str = id_to_label_map.get(idx)

        if not label_str:
            continue

        colon_count = label_str.count(':')
        
        if colon_count == 1:
            narrative_indices.append(idx)
        elif colon_count == 2:
            subnarrative_indices.append(idx)

    # Sort the lists for consistency
    narrative_indices.sort()
    subnarrative_indices.sort()
    
    print(f"Found {len(narrative_indices)} narrative-level indices and {len(subnarrative_indices)} sub-narrative-level indices.")

    return {
        'narrative_indices': narrative_indices,
        'subnarrative_indices': subnarrative_indices
    }