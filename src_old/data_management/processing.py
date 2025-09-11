
import pandas as pd
from src.data_management.preprocessor import binarize_labels

def process_dataframe_for_training(df: pd.DataFrame, label_to_id: dict, all_ids: list) -> pd.DataFrame:
    """
    Processes a DataFrame to create a training-ready format with binarized labels.

    Args:
        df (pd.DataFrame): The input DataFrame with 'text', 'narratives', and 'subnarratives' columns.
        label_to_id (dict): A dictionary mapping labels to their corresponding IDs.
        all_ids (list): A list of all possible label IDs.

    Returns:
        pd.DataFrame: The processed DataFrame with a 'labels' column containing the binarized label vectors.
    """
    
    def map_labels_to_ids(labels, mapping):
        return [mapping.get(label) for label in labels if mapping.get(label) is not None]

    df['narrative_ids'] = df['narratives'].apply(map_labels_to_ids, args=(label_to_id,))
    df['subnarrative_ids'] = df['subnarratives'].apply(map_labels_to_ids, args=(label_to_id,))

    df['labels'] = df.apply(lambda row: row['narrative_ids'] + row['subnarrative_ids'], axis=1)
    df['labels'] = df['labels'].apply(lambda x: binarize_labels(x, all_ids))
    
    return df
