import numpy as np
import pandas as pd
import torch
import os

def read_text_file(article_id, lang, base_path = 'data', docs_folder='raw-documents'):
    file_path = f"{base_path}/{lang}/{docs_folder}/{article_id}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def load_all_annotations_to_df(base_path = 'data', lang_folders = ['BG', 'EN', 'HI', 'PT', 'RU'], annotation_file_name = 'subtask-2-annotations.txt', docs_folder='raw-documents'):
    all_dfs = []
    for lang in lang_folders:
        file_path = f"{base_path}/{lang}/{annotation_file_name}"
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'narratives', 'subnarratives'])
            df['narratives'] = df['narratives'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            df['subnarratives'] = df['subnarratives'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            df['language'] = lang
            df['text'] = df['id'].apply(lambda x: read_text_file(x, lang, base_path, docs_folder=docs_folder))
            df = df[['id', 'text', 'narratives', 'subnarratives', 'language']]
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Empty file: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        print("No data frames were created. Please check the file paths and contents.")
        return pd.DataFrame(columns=['id', 'narratives', 'subnarratives', 'language'])


def load_ids_to_df(annotations_df, label_to_id):
    annotations_df['narrative_ids'] = annotations_df['narratives'].apply(lambda x: [label_to_id.get(n, -1) for n in x])
    annotations_df['subnarrative_ids'] = annotations_df['subnarratives'].apply(lambda x: [label_to_id.get(sn, -1) for sn in x])
    #make the ids unique
    annotations_df['narrative_ids'] = annotations_df['narrative_ids'].apply(lambda x: list(set(x)))
    annotations_df['subnarrative_ids'] = annotations_df['subnarrative_ids'].apply(lambda x: list(set(x)))
    return annotations_df[['id', 'text', 'narratives', 'subnarratives', 'narrative_ids', 'subnarrative_ids', 'language']]

def _resolve_path(dataframe, base_path):
    """Resolve a dataframe path: if `dataframe` already points to an existing file or is absolute,
    return it as-is; otherwise join it with `base_path`.
    This makes callers resilient to passing either 'phase0.parquet' or 'data/processed/phase0.parquet'.
    """
    # If it's an absolute path or already exists relative to cwd, use it directly
    if os.path.isabs(dataframe) or os.path.exists(dataframe):
        return dataframe

    # Otherwise try joining with base_path
    candidate = os.path.join(base_path, dataframe)
    return candidate


def load_tokenized_df(dataframe, base_path = 'data/processed'):
    path = _resolve_path(dataframe, base_path)
    df = pd.read_parquet(path)
    df['input_ids_pt'] = df['input_ids_list'].apply(lambda x: torch.tensor(x, dtype=torch.long)) # type: ignore
    df['attention_mask_pt'] = df['attention_mask_list'].apply(lambda x: torch.tensor(x, dtype=torch.long)) # type: ignore
    df['labels_pt'] = df['labels'].apply(lambda x: torch.tensor(x, dtype=torch.float)) # type: ignore
    
    return df

def load_labeled_df(dataframe, base_path = 'data/processed'):
    path = _resolve_path(dataframe, base_path)
    df = pd.read_parquet(path)
    df['narrative_ids'] = df['narrative_ids'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df['subnarrative_ids'] = df['subnarrative_ids'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

if __name__ == "__main__":
    from label_parser import parse_json_for_narratives_subnarratives, create_label_mappings
    import os

    # Path to taxonomy JSON
    taxonomy_path = os.path.join('data', 'taxonomy.json')
    narratives, subnarratives = parse_json_for_narratives_subnarratives(taxonomy_path)
    label_to_id, id_to_label, narrative_to_subnarrative_ids = create_label_mappings(narratives, subnarratives)

    # Load annotations DataFrame
    df = load_all_annotations_to_df()
    print("Loaded annotations DataFrame:")
    print(df.head())

    # Map labels to IDs
    df_with_ids = load_ids_to_df(df, label_to_id)
    print("\nDataFrame with narrative and subnarrative IDs:")
    print(df_with_ids.head())

