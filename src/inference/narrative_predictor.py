import torch
import numpy as np
import pandas as pd


class NarrativePredictor:
    def __init__(self, model_path, tokenizer_name, label_maps, correction_strategy='add_other', device=None):
        """
        Initialize the NarrativePredictor.

        Args:
            model_path (str): Path to the model weights.
            tokenizer_name (str): Name or path of the tokenizer.
            label_maps (dict): Dictionary with 'label2id', 'id2label', and 'parent_child_pairs'.
            correction_strategy (str): Either 'add_other' or 'prune'.
            device (str or torch.device, optional): Device to use.
        """
        print("initializing the Narrative Predictor...")

        if correction_strategy not in ['add_other', 'prune']:
            raise ValueError("correction_strategy must be either 'add_other' or 'prune'")
        self.correction_strategy = correction_strategy

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
            tokenizer_name,
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
    
    def _apply_final_correction(self, narratives, subnarratives):
        """Applies the chosen strategy for childless narratives."""
        if not narratives:
            return [], []

        parents_with_children = {
            ":".join(sub_str.split(":")[:-1]).strip() for sub_str in subnarratives
        }

        if self.correction_strategy == 'add_other':
            childless_narratives = [n_str for n_str in narratives if n_str not in parents_with_children]
            for n_str in childless_narratives:
                other_sub_narr = f"{n_str}: Other"
                # Check if this "Other" is a valid label before adding
                if self.label2id.get(other_sub_narr) is not None:
                    if other_sub_narr not in subnarratives:
                        subnarratives.append(other_sub_narr)
            return narratives, subnarratives

        elif self.correction_strategy == 'prune':
            pruned_narratives = [n_str for n_str in narratives if n_str in parents_with_children]
            return pruned_narratives, subnarratives
        
        return narratives, subnarratives # Should not be reached
    
    def predict(self, text: str):
        """Predicts narratives for a single text."""
        # The logic is the same as for a batch of one
        results = self.predict_batch([text])
        return results[0] # Return the results for the single text
    
    def predict_batch(self, texts: list, file_path=None):
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
            prediction_vector = binary_predictions[i]
            positive_indices = np.where(prediction_vector == 1)[0]
            
            # Initial categorization
            narratives = []
            subnarratives = []
            for idx in positive_indices:
                label_str = self.id2label.get(int(idx))
                if not label_str: continue
                
                if label_str.count(':') == 1:
                    narratives.append(label_str)
                elif label_str.count(':') == 2:
                    subnarratives.append(label_str)
            
            # Apply the final correction logic here
            final_narratives, final_subnarratives = self._apply_final_correction(narratives, subnarratives)

            # If both are empty, put 'Other' in both
            if not final_narratives and not final_subnarratives:
                final_narratives = ['Other']
                final_subnarratives = ['Other']
            
            results.append({
                "narratives": sorted(final_narratives),
                "subnarratives": sorted(final_subnarratives)
            })
        if file_path:
            self._save_prediction_to_file(results, file_path)
            print(f"Predictions saved to {file_path}")
            
        return results
    
    def _save_prediction_to_file(self, predictions, file_path):
        """
        Saves the predictions to a CSV file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("narratives;subnarratives\n")
            for pred in predictions:
                narratives_str = ";".join(pred.get("narratives", []))
                subnarratives_str = ";".join(pred.get("subnarratives", []))
                f.write(f"{narratives_str};{subnarratives_str}\n")