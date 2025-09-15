import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm.auto import tqdm

# --- Configuration ---
# Path to your saved fine-tuned model directory
MODEL_PATH = "./final_semantic_classifier"
# Path to the label embeddings file created during preprocessing
LABEL_EMBEDDINGS_PATH = "embeddings/label_embeddings.pt"
# Path to the label mappings file
LABEL_MAPPINGS_PATH = "label_mappings.json"
# Path to the folder containing your test documents
TEST_DATA_DIR = "testset/EN/subtask-2-documents"
# The decision threshold for classifying a label as positive
# Let's use the optimal one we found during training!
PREDICTION_THRESHOLD = 0.43

# --- Step 1: Reload the Custom Model Definition ---
# This class definition must be available when loading the model.
class SemanticClassifier(nn.Module):
    def __init__(self, model_name: str, label_embeddings: torch.Tensor):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.register_buffer("label_embeddings", label_embeddings)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = self.dropout(text_embedding)
        text_embedding_norm = F.normalize(text_embedding, p=2, dim=1)
        label_embeddings_norm = F.normalize(self.label_embeddings, p=2, dim=1)
        logits = torch.matmul(text_embedding_norm, label_embeddings_norm.t())
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return {"loss": loss, "logits": logits}

# --- Step 2: Load Model, Tokenizer, and Label Information ---
print("--- Loading model, tokenizer, and label info ---")

# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the label embeddings and mappings
label_embeddings = torch.load(LABEL_EMBEDDINGS_PATH)
with open(LABEL_MAPPINGS_PATH, 'r') as f:
    label_mappings = json.load(f)
id2label = {int(k): v for k, v in label_mappings['id2label'].items()}

# Load the tokenizer from the saved model directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Instantiate the model with the correct architecture and embeddings
model = SemanticClassifier(
    model_name=MODEL_PATH,
    label_embeddings=label_embeddings
)
# The Trainer saves the state_dict, so we need to load it
# Note: The model is already initialized with the pre-trained weights from MODEL_PATH.
# The trainer.save_model() saves the fine-tuned weights in pytorch_model.bin,
# which AutoModel.from_pretrained() loads automatically. So, no explicit state_dict loading is needed.

model.to(device) # Move the model to the GPU if available
model.eval()     # Set the model to evaluation mode (disables dropout)

print("✅ Model and tokenizer loaded successfully.")

# --- Step 3: Define the Prediction Function ---
def predict(text: str):
    """
    Takes a raw text string and returns the predicted labels and their probabilities.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt" # Return PyTorch tensors
    )

    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad(): # Disable gradient calculations for efficiency
        outputs = model(**inputs)
        logits = outputs['logits']

    # Get probabilities by applying sigmoid
    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()

    # Get binary predictions based on the threshold
    predictions = (probs >= PREDICTION_THRESHOLD).astype(int)

    # Map predictions to label names
    predicted_labels = []
    for i, val in enumerate(predictions):
        if val == 1:
            label_name = id2label[i]
            # Include the probability for context
            predicted_labels.append({"label": label_name, "probability": round(float(probs[i]), 4)})
            
    return predicted_labels

# --- Step 4: Run Inference on the Test Directory ---
print(f"\n--- Running inference on files in '{TEST_DATA_DIR}' ---")

# Get a list of all .txt files in the directory
try:
    test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.txt')]
    if not test_files:
        print(f"Warning: No .txt files found in '{TEST_DATA_DIR}'")
except FileNotFoundError:
    print(f"Error: Directory not found: '{TEST_DATA_DIR}'")
    test_files = []


# Process each file
for filename in tqdm(test_files, desc="Classifying documents"):
    file_path = os.path.join(TEST_DATA_DIR, filename)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Get predictions for the file content
    predicted_labels = predict(content)
    
    # Print the results for the current file
    print(f"\n--- Results for: {filename} ---")
    if predicted_labels:
        # Sort by probability for easier reading
        predicted_labels.sort(key=lambda x: x['probability'], reverse=True)
        for item in predicted_labels:
            print(f"  - Label: {item['label']} (Confidence: {item['probability']:.2f})")
    else:
        print("  - No labels predicted above the threshold.")
    print("-" * (len(filename) + 20))