import torch
import numpy as np
import pandas as pd


class NarrativePredictor:
    def __init__(self, model_path, tokenizer_name, label_maps, device=None):
        
        print("initializing the Narrative Predictor...")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        else:
            device = torch.device(device)
            print(f"Using specified device: {device}")
        
        self.device = device
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        from transformers import AutoModelForSequenceClassification
        self.label2id = label_maps['label2id']
        self.id2label = label_maps['id2label']
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type='multi_label_classification'
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode permanently
        
        self.parent_child_pairs = label_maps['parent_child_pairs']
        self.threshold = 0.5 # Default threshold, can be updated
        print("Predictor initialized and ready.")
        
    
    def set_threshold(self, threshold):
        """
        Set the threshold for binary classification.
        """
        self.threshold = threshold
        print(f"Threshold set to: {self.threshold}")
        
    def _process_predictions(self, probabilities):
        """Converts probabilities to binary predictions and applies hierarchical correction."""
        binary_preds = (probabilities > self.threshold).astype(int)
        
        # Apply hierarchical correction
        for sub_id, narr_id in self.parent_child_pairs:
            inconsistent_mask = (binary_preds[:, sub_id] == 1) & (binary_preds[:, narr_id] == 0)
            binary_preds[inconsistent_mask, sub_id] = 0
            
        return binary_preds
    
    def predict(self, text: str):
        """Predicts narratives for a single text."""
        # The logic is the same as for a batch of one
        results = self.predict_batch([text])
        return results[0] # Return the results for the single text
    
    def predict_batch(self, texts: list):
        """
        Predicts narratives for a batch of texts.
        """
        inputs = self.tokenizer(
            texts,
            padding=True, # Pad to the longest sequence in the batch
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().numpy()

        binary_predictions = self._process_predictions(probabilities)
        
        # Convert binary predictions back to label strings
        results = []
        for i in range(len(texts)):
            narrative_ids = [idx for idx, val in enumerate(binary_predictions[i]) if val == 1 and self.id2label[idx].count(':') == 1]
            subnarrative_ids = [idx for idx, val in enumerate(binary_predictions[i]) if val == 1 and self.id2label[idx].count(':') == 2]

            narratives = [self.id2label[idx] for idx in narrative_ids]
            subnarratives = [self.id2label[idx] for idx in subnarrative_ids]
            
            results.append({
                "narratives": narratives,
                "subnarratives": subnarratives
            })
            
        return results