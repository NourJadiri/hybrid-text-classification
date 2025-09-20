#!/usr/bin/env python3
"""
Script to push the mDeberta hierarchical classification model to Hugging Face Hub
"""

import os
import json
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoConfig

def push_model_to_hub():
    # Configuration
    model_path = "models/microsoft/mdeberta-v3-base_narratives_classifier_hierarchical"
    repo_name = "mdeberta-v3-base-narratives-classifier-hierarchical"  # You can change this
    username = "AWCO"  # Your authenticated HF username
    
    # Full repository ID
    repo_id = f"{username}/{repo_name}"
    
    print(f"🚀 Pushing model to: https://huggingface.co/{repo_id}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path {model_path} does not exist!")
        return
    
    # Initialize HF API
    api = HfApi()
    
    try:
        # Create repository (will not fail if it already exists)
        print(f"📝 Creating repository {repo_id}...")
        create_repo(
            repo_id=repo_id,
            token=None,  # Uses stored token
            private=False,  # Set to True if you want a private repo
            exist_ok=True
        )
        
        # Create a model card (README.md)
        readme_content = f"""---
library_name: transformers
license: mit
base_model: microsoft/mdeberta-v3-base
tags:
- text-classification
- hierarchical-classification
- climate-change
- ukraine-russia-war
- narrative-classification
- multilingual
datasets:
- custom
language:
- en
- bg
- hi
- pt
- ru
metrics:
- f1
- precision
- recall
pipeline_tag: text-classification
---

# mDeBERTa v3 Base - Hierarchical Narratives Classifier

This model is a fine-tuned version of [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) for hierarchical text classification of narratives related to Climate Change and Ukraine-Russia War.

## Model Description

This model performs hierarchical classification on text documents to identify:
1. **Main Narrative Categories**: Climate Change (CC) and Ukraine-Russia War (URW) related narratives
2. **Sub-Narrative Classifications**: Detailed subcategories within each main narrative

## Supported Languages

The model has been trained and tested on multiple languages:
- English (EN)
- Bulgarian (BG) 
- Hindi (HI)
- Portuguese (PT)
- Russian (RU)

## Usage

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example usage for inference
text = "Your text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    # Process outputs according to your inference pipeline
```

## Training Data

The model was trained on a custom dataset containing narratives related to:
- Climate Change discussions and misinformation
- Ukraine-Russia War narratives and propaganda

## Performance

The model achieves strong performance across multiple languages for both narrative-level and sub-narrative level classification tasks.

## Model Files

- `config.json`: Model configuration
- `model.safetensors`: Model weights in SafeTensors format
- `tokenizer.json`: Tokenizer configuration
- `label_mappings_hierarchical.json`: Hierarchical label mappings for classification

## Citation

If you use this model in your research, please cite accordingly.

## License

This model is released under the MIT License.
"""
        
        # Write README.md
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"📄 Created README.md")
        
        # Upload all files from the model directory
        print(f"📤 Uploading files to {repo_id}...")
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=None,  # Uses stored token
            commit_message="Upload mDeBERTa hierarchical narratives classifier"
        )
        
        print(f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        print(f"🎉 Your model is now available on the Hugging Face Hub!")
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        print("💡 Make sure you're logged in with 'huggingface-cli login' and have write permissions")

if __name__ == "__main__":
    push_model_to_hub()