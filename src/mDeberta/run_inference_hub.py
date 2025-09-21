import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download

# Import the custom model class and the utility function
from multihead_deberta import MultiHeadDebertaForHierarchicalClassification, sanitize_name

def run_inference_from_hub(model_name, input_dir, output_file, threshold=0.5):
    """
    Runs inference on a directory of text files using a trained hierarchical model from HuggingFace Hub.

    Args:
        model_name (str): HuggingFace model name (e.g., "AWCO/mdeberta-v3-base-narratives-classifier-hierarchical").
        input_dir (str): Path to the directory containing input .txt files.
        output_file (str): Path to the output .tsv file.
        threshold (float): Classification threshold.
    """
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Step 1: Setting up configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Download and Load Model from HuggingFace Hub
    # -------------------------------------------------------------------------
    print(f"Step 2: Downloading model '{model_name}' from HuggingFace Hub...")
    
    # Download the entire model directory
    model_path = snapshot_download(repo_id=model_name)
    print(f"Model downloaded to: {model_path}")

    # Load training configuration to get the best threshold if available
    try:
        with open(os.path.join(model_path, 'training_config.json'), 'r') as f:
            training_config = json.load(f)
            threshold = training_config.get('best_threshold', threshold)
            print(f"Loaded best threshold from training: {threshold:.3f}")
    except FileNotFoundError:
        print(f"Warning: training_config.json not found. Using threshold: {threshold}")

    # Load hierarchical label mappings
    with open(os.path.join(model_path, 'label_mappings_hierarchical.json'), 'r') as f:
        label_mappings = json.load(f)
        parent_label2id = label_mappings['parent_label2id']
        # JSON keys are strings, so convert parent_id back to int for indexing
        parent_id2label = {int(k): v for k, v in label_mappings['parent_id2label'].items()}
        child_label_maps = label_mappings['child_label_maps']
    print("Loaded hierarchical label mappings.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the custom model
    model = MultiHeadDebertaForHierarchicalClassification.from_pretrained(
        model_path,
        parent_label2id=parent_label2id,
        child_label_maps=child_label_maps
    )
    model.to(device)
    model.eval() # Set the model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # -------------------------------------------------------------------------
    # 3. Find and Process Input Files
    # -------------------------------------------------------------------------
    print("Step 3: Finding and processing input text files...")
    try:
        text_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        if not text_files:
            print(f"Error: No .txt files found in directory '{input_dir}'")
            return
        print(f"Found {len(text_files)} text files to process.")
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    all_predictions = []
    for filename in tqdm(text_files, desc="Inferring"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Tokenize the text for the model
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512 # Use the same max_length as in training
        ).to(device)

        # ---------------------------------------------------------------------
        # 4. Run Inference and Decode Predictions
        # ---------------------------------------------------------------------
        with torch.no_grad(): # Disable gradient calculation for efficiency
            outputs = model(**inputs)

        # Unpack the logits from the model's output dictionary
        parent_logits = outputs['parent_logits']
        child_logits_dict = outputs['child_logits']

        # Get parent predictions
        parent_probs = torch.sigmoid(parent_logits).cpu().numpy().flatten()
        parent_preds_indices = (parent_probs >= threshold).astype(int)

        predicted_parents = []
        predicted_subnarratives = []

        # Loop through parent predictions to decode them
        for i, is_present in enumerate(parent_preds_indices):
            if is_present:
                parent_name = parent_id2label[i]
                predicted_parents.append(parent_name)

                # If a parent is predicted, check for its children
                if parent_name in child_label_maps:
                    safe_key = sanitize_name(parent_name)
                    child_logits = child_logits_dict[safe_key]
                    child_probs = torch.sigmoid(child_logits).cpu().numpy().flatten()
                    child_preds_indices = (child_probs >= threshold).astype(int)
                    
                    # Get the specific id->label map for this parent's children
                    child_id2label = child_label_maps[parent_name]['id2label']

                    # Loop through child predictions to decode them
                    for j, child_is_present in enumerate(child_preds_indices):
                        if child_is_present:
                            # JSON keys are strings, so we access with str(j)
                            child_name = child_id2label[str(j)]
                            # child_name already contains the full hierarchical path
                            predicted_subnarratives.append(child_name)

        # Format the predictions into semicolon-separated strings
        if not predicted_parents:
            narratives_str = "Other"
            subnarratives_str = "Other"
        else:
            narratives_str = ";".join(sorted(predicted_parents))
            subnarratives_str = ";".join(sorted(predicted_subnarratives))

        all_predictions.append((filename, narratives_str, subnarratives_str))

    # -------------------------------------------------------------------------
    # 5. Write Output File
    # -------------------------------------------------------------------------
    print(f"Step 4: Writing {len(all_predictions)} predictions to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename, narratives, subnarratives in all_predictions:
            f.write(f"{filename}\t{narratives}\t{subnarratives}\n")

    print("Inference complete!")


def run_inference_all_languages(model_name, devset_base_dir, output_dir, threshold=0.5):
    """
    Run inference on all languages in the devset.
    
    Args:
        model_name (str): HuggingFace model name.
        devset_base_dir (str): Base directory containing language folders.
        output_dir (str): Directory to save output files.
        threshold (float): Classification threshold.
    """
    languages = ['BG', 'EN', 'HI', 'PT', 'RU']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for lang in languages:
        print(f"\n{'='*60}")
        print(f"Processing language: {lang}")
        print(f"{'='*60}")
        
        input_dir = os.path.join(devset_base_dir, lang, 'subtask-2-documents')
        output_file = os.path.join(output_dir, f"{lang}_predictions.tsv")
        
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found for {lang}: {input_dir}")
            continue
            
        run_inference_from_hub(model_name, input_dir, output_file, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with a hierarchical text classification model from HuggingFace Hub.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="AWCO/mdeberta-v3-base-narratives-classifier-hierarchical",
        help="HuggingFace model name (default: AWCO/mdeberta-v3-base-narratives-classifier-hierarchical)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory containing the input .txt files (for single language inference)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.tsv",
        help="Path to the output file where predictions will be saved (default: predictions.tsv)."
    )
    parser.add_argument(
        "--devset_dir",
        type=str,
        default="../../devset",
        help="Path to the devset directory containing all language folders (default: ../../devset)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../results/hub_inference",
        help="Directory to save output files for all languages (default: ../../results/hub_inference)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)."
    )
    parser.add_argument(
        "--all_languages",
        action="store_true",
        help="Run inference on all languages in the devset."
    )
    
    args = parser.parse_args()

    if args.all_languages:
        run_inference_all_languages(args.model_name, args.devset_dir, args.output_dir, args.threshold)
    elif args.input_dir:
        run_inference_from_hub(args.model_name, args.input_dir, args.output_file, args.threshold)
    else:
        print("Error: Please specify either --input_dir for single language inference or --all_languages for all languages.")