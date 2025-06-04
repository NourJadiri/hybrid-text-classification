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

    