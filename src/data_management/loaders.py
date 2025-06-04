import pandas as pd

def read_text_file(article_id, lang, base_path = 'data'):
    file_path = f"{base_path}/{lang}/raw-documents/{article_id}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def load_all_annotations_to_df(base_path = 'data', lang_folders = ['BG', 'EN', 'HI', 'PT', 'RU'], annotation_file_name = 'subtask-2-annotations.txt'):
    all_dfs = []
    for lang in lang_folders:
        file_path = f"{base_path}/{lang}/{annotation_file_name}"
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'narratives', 'subnarratives'])
            df['narratives'] = df['narratives'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            df['subnarratives'] = df['subnarratives'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
            df['language'] = lang
            df['text'] = df['id'].apply(lambda x: read_text_file(x, lang, base_path))
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

