
import pandas as pd
import torch
from torch.utils.data import Dataset

class NarrativeClassificationDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512):
        """
        Initializes the dataset with a DataFrame, tokenizer, and maximum sequence length.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'text' and 'label' columns.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
            max_length (int): Maximum length of the tokenized sequences.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx : int):
       
        data_row = self.dataframe.iloc[idx]
        text = str(data_row['text'])
        labels = torch.tensor(data_row['labels'], dtype=torch.float)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding="max_length",       # Pad all sentences to max_length
            truncation=True,            # Truncate sentences longer than max_length
            return_attention_mask=True,
            return_tensors='pt',        # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),  
            'attention_mask': encoding['attention_mask'].flatten(), 
            'labels': labels
        }
        