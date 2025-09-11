import os
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from transformers import AutoTokenizer

from src.data_management.loaders import load_all_annotations_to_df
from src.data_management.label_parser import parse_json_for_narratives_subnarratives, create_label_mappings
from src.data_management.preprocessor import binarize_labels
from src.data_management.datasets import NarrativeClassificationDataset

def prepare_dataframes(data_folder, docs_folder='raw-documents'):
    """
    Loads and preprocesses data into train, validation, and test DataFrames.

    Args:
        data_folder (str): Path to the root data folder (e.g., 'data').
        docs_folder (str, optional): Name of the folder containing the raw documents. Defaults to 'raw-documents'.

    Returns:
        tuple: A tuple containing:
            - train_df (DataFrame): The training data.
            - val_df (DataFrame): The validation data.
            - test_df (DataFrame): The test data.
            - id_to_label (dict): Mapping from label ID to label string.
            - label_to_id (dict): Mapping from label string to label ID.
            - parent_child_pairs (list): A list of tuples mapping subnarrative IDs to narrative IDs.
    """
    # --- 1. Load Annotations and Taxonomy ---
    print("Loading annotations and taxonomy...")
    taxonomy_path = os.path.join(data_folder, 'taxonomy.json')
    
    # It's assumed that load_all_annotations_to_df knows to look for language subfolders (EN, BG, etc.)
    # inside the provided data_folder.
    annotations_df = load_all_annotations_to_df(base_path=data_folder, docs_folder=docs_folder)
    
    narratives, subnarratives = parse_json_for_narratives_subnarratives(taxonomy_path)
    label_to_id, id_to_label, narrative_to_subnarrative_ids = create_label_mappings(narratives, subnarratives)
    all_ids = list(id_to_label.keys())

    # --- 2. Map Labels to IDs and Binarize ---
    print("Mapping labels to IDs and creating binarized vectors...")
    def map_labels_to_ids(labels, mapping):
        return [mapping[label] for label in labels if label in mapping]

    annotations_df['narrative_ids'] = annotations_df['narratives'].apply(map_labels_to_ids, args=(label_to_id,))
    annotations_df['subnarrative_ids'] = annotations_df['subnarratives'].apply(map_labels_to_ids, args=(label_to_id,))

    annotations_df['labels'] = annotations_df.apply(lambda row: row['narrative_ids'] + row['subnarrative_ids'], axis=1)
    annotations_df['labels'] = annotations_df['labels'].apply(lambda x: binarize_labels(x, all_ids))

    # --- 3. Split Dataset ---
    print("Splitting dataset into train, validation, and test sets...")
    df = annotations_df
    X = df.index.to_numpy().reshape(-1, 1)
    y = np.array(df['labels'].tolist())

    # 80% for training/validation, 20% for testing
    train_val_indices, _, test_indices, _ = iterative_train_test_split(X, y, test_size=0.2)
    
    # Of the 80%, 75% for training, 25% for validation (which is 60/20 of the total)
    train_indices, _, val_indices, _ = iterative_train_test_split(
        train_val_indices, y[train_val_indices.flatten()], test_size=0.25
    )

    train_df = df.loc[train_indices.flatten()].reset_index(drop=True)
    val_df = df.loc[val_indices.flatten()].reset_index(drop=True)
    test_df = df.loc[test_indices.flatten()].reset_index(drop=True)
    
    print(f"Dataset split sizes: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    # --- 4. Prepare Hierarchical Training Parameters ---
    sub_to_narr_id_map = {sub_id: narr_id for narr_id, sub_ids in narrative_to_subnarrative_ids.items() for sub_id in sub_ids}
    parent_child_pairs = list(sub_to_narr_id_map.items())
    
    return (
        train_df,
        val_df,
        test_df,
        id_to_label,
        label_to_id,
        parent_child_pairs
    )


def prepare_datasets(data_folder, model_name='xlm-roberta-base', max_length=512, docs_folder='raw-documents'):
    """
    Loads, preprocesses, and splits the data for training.

    This function encapsulates the steps from the data loading and preprocessing notebooks:
    1. Loads annotations from the specified data folder.
    2. Parses the taxonomy to create label mappings.
    3. Maps textual labels to integer IDs.
    4. Creates binarized label vectors.
    5. Splits the data into training, validation, and test sets using iterative stratification.
    6. Creates PyTorch-compatible datasets for each split.

    Args:
        data_folder (str): Path to the root data folder (e.g., 'data').
        model_name (str, optional): Name of the transformer model to use for tokenization. 
                                    Defaults to 'xlm-roberta-base'.
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        docs_folder (str, optional): Name of the folder containing the raw documents. Defaults to 'raw-documents'.

    Returns:
        tuple: A tuple containing:
            - train_dataset (NarrativeClassificationDataset): The training dataset.
            - val_dataset (NarrativeClassificationDataset): The validation dataset.
            - test_dataset (NarrativeClassificationDataset): The test dataset.
            - tokenizer (AutoTokenizer): The tokenizer instance.
            - id_to_label (dict): Mapping from label ID to label string.
            - parent_child_pairs (list): A list of tuples mapping subnarrative IDs to narrative IDs.
            - num_total_labels (int): The total number of unique labels.
    """
    
    train_df, val_df, test_df, id_to_label, label_to_id, parent_child_pairs = prepare_dataframes(
        data_folder, docs_folder
    )
    
    num_total_labels = len(id_to_label)

    # --- 5. Create Tokenizer and PyTorch Datasets ---
    print(f"Loading tokenizer ('{model_name}') and creating PyTorch datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NarrativeClassificationDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = NarrativeClassificationDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = NarrativeClassificationDataset(test_df, tokenizer, max_length=max_length)
    
    print("Data preparation complete.")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        id_to_label,
        label_to_id,
        parent_child_pairs,
        num_total_labels,
    )


