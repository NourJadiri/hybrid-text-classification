#!/usr/bin/env python3
# src/scripts/batch_inference.py

import argparse
import os
import sys
import glob
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from utils.prompt_template import (
    create_comprehensive_prompt_template,
    create_simple_prompt_template,
    create_constrained_prompt_template,
)

def load_model(model_path, max_seq_length=4096, chat_template="qwen3-instruct"):
    """Load the model and tokenizer using unsloth (works for both vanilla and fine-tuned models)."""
    print(f"Loading model from: {model_path}")
    
    # Load model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    # Configure chat template
    tokenizer = get_chat_template(
        tokenizer=tokenizer,
        chat_template=chat_template
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def predict_narratives(text, model, tokenizer, prompt_type: str = "constrained", return_prompt: bool = False):
    """Predict narratives for the given text.

    prompt_type: one of 'constrained', 'simple' or 'comprehensive'.
    Defaults to 'constrained'.
    """

    # Choose prompt template based on prompt_type
    if prompt_type == "simple":
        prompt_template = create_simple_prompt_template()
    elif prompt_type == "comprehensive":
        prompt_template = create_comprehensive_prompt_template()
    else:
        # default to constrained
        prompt_template = create_constrained_prompt_template()
    
    # Format the prompt
    user_content = prompt_template.format(text=text)
    
    messages = [
        {"role": "user", "content": user_content},
    ]
    
    # Tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=2048,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    # Decode the response
    response = tokenizer.batch_decode(outputs)
    full_response = response[0]
    
    # Extract the assistant's response
    if "<|im_start|>assistant\n" in full_response:
        assistant_response = full_response.split("<|im_start|>assistant\n")[1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.replace("<|im_end|>", "").strip()
        return (assistant_response, user_content) if return_prompt else assistant_response
    
    return (full_response, user_content) if return_prompt else full_response

def parse_narratives(response):
    """Parse the model response to extract narratives."""
    import re
    
    # Look for content within brackets
    bracket_pattern = r'\[(.*?)\]'
    matches = re.findall(bracket_pattern, response)
    
    if matches:
        content = matches[0].strip()
        if content.lower() == 'other' or content.lower() == 'none':
            return ['Other']
        
        # Split by semicolon and clean
        narratives = [n.strip() for n in content.split(';') if n.strip()]
        return narratives if narratives else ['Other']
    
    # If no brackets found, return Other
    return ['Other']

def find_text_files(input_path: str) -> List[str]:
    """Find all text files in the given path."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Find all .txt files in the directory
        txt_files = glob.glob(os.path.join(input_path, "*.txt"))
        if not txt_files:
            print(f"Warning: No .txt files found in directory {input_path}")
        return sorted(txt_files)
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")

def process_text_files(text_files: List[str], model, tokenizer, prompt_type: str, full_output: bool) -> List[Dict]:
    """Process all text files and return results."""
    results = []
    
    total_files = len(text_files)
    print(f"Processing {total_files} text files...")
    
    for i, file_path in enumerate(text_files, 1):
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            # Skip empty files
            if not text_content:
                print(f"Warning: Empty file {file_path}, skipping...")
                continue
            
            # Get filename without path
            filename = os.path.basename(file_path)
            
            print(f"[{i}/{total_files}] Processing: {filename}")
            
            # Get prediction
            pred = predict_narratives(text_content, model, tokenizer, prompt_type, return_prompt=full_output)
            if full_output:
                raw_response, used_prompt = pred
            else:
                raw_response = pred
                used_prompt = None
            detected_narratives = parse_narratives(raw_response)
            
            # Store result
            result = {
                'filename': filename,
                'detected_narratives': detected_narratives,
                'raw_response': raw_response
            }
            if full_output:
                result['prompt'] = used_prompt
                result['text'] = text_content
            results.append(result)
            
            # Print progress
            narrative_str = ';'.join(detected_narratives)
            print(f"  -> Detected: {narrative_str}")
            if full_output:
                print("  --- Full Output Start ---")
                print(f"  [PROMPT]\n{used_prompt}")
                print(f"  [TEXT]\n{text_content}")
                print(f"  [MODEL OUTPUT]\n{raw_response}")
                print("  --- Full Output End ---")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Add error result
            results.append({
                'filename': os.path.basename(file_path),
                'detected_narratives': ['Other'],
                'raw_response': f"Error: {str(e)}"
            })
    
    return results

def write_annotation_file(results: List[Dict], output_path: str):
    """Write results to annotation file in the required format."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            filename = result['filename']
            narratives = result['detected_narratives']
            
            # Format narratives as semicolon-separated string
            narrative_str = ';'.join(narratives)
            
            # For the detailed column, we'll use the same narratives for now
            # In a real implementation, you might want to add subcategory details
            detailed_str = narrative_str
            
            # Write in TSV format: filename \t narratives \t detailed_narratives
            f.write(f"{filename}\t{narrative_str}\t{detailed_str}\n")
    
    print(f"Results written to: {output_path}")

def single_text_inference(text_input: str, model, tokenizer, prompt_type: str, full_output: bool):
    """Process a single text input (for backward compatibility)."""
    
    print("="*60)
    print("QWEN NARRATIVE DETECTION - SINGLE TEXT")
    print("="*60)
    print(f"Input Text: {text_input[:200]}..." if len(text_input) > 200 else f"Input Text: {text_input}")
    print("-"*60)
    
    # Get prediction
    pred = predict_narratives(text_input, model, tokenizer, prompt_type, return_prompt=full_output)
    if full_output:
        raw_response, used_prompt = pred
    else:
        raw_response = pred
        used_prompt = None
    detected_narratives = parse_narratives(raw_response)
    
    # Output results
    print(f"Raw Response: {raw_response}")
    if full_output:
        print("-"*60)
        print("FULL OUTPUT")
        print("[PROMPT]")
        print(used_prompt)
        print("[TEXT]")
        print(text_input)
        print("[MODEL OUTPUT]")
        print(raw_response)
    print(f"Detected Narratives: {detected_narratives}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on text files with finetuned Qwen model.")
    
    # Input options - make mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Single text to analyze for narratives.")
    input_group.add_argument("--file", type=str, help="Path to single text file to analyze.")
    input_group.add_argument("--folder", type=str, help="Path to folder containing text files to process.")
    
    # Model and processing options
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model.")
    parser.add_argument("--prompt_type", type=str, choices=["constrained", "simple", "comprehensive"], 
                        default="constrained", help="Type of prompt template to use.")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="The maximum sequence length of the model.")
    parser.add_argument("--chat_template", type=str, default="qwen3-instruct",
                        help="Chat template to use.")
    
    # Output options
    parser.add_argument("--output", type=str, 
                        help="Output file path for batch results (TSV format). If not specified, uses input folder name.")
    parser.add_argument("--full_output", action="store_true",
                        help="Include full prompt and input text in outputs and logs. If set, batch TSV will include additional columns: prompt, text, model_output.")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, 
                                  max_seq_length=args.max_seq_length,
                                  chat_template=args.chat_template)
    print("Model loaded successfully!")
    
    # Handle different input types
    if args.text:
        # Single text input
        single_text_inference(args.text, model, tokenizer, args.prompt_type, args.full_output)
        
    elif args.file:
        # Single file input
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            single_text_inference(text_content, model, tokenizer, args.prompt_type, args.full_output)
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}")
            return
            
    elif args.folder:
        # Batch processing
        print("="*60)
        print("QWEN NARRATIVE DETECTION - BATCH PROCESSING")
        print("="*60)
        print(f"Input folder: {args.folder}")
        print(f"Model: {args.model_path}")
        print(f"Prompt type: {args.prompt_type}")
        print("-"*60)
        
        try:
            # Find all text files
            text_files = find_text_files(args.folder)
            
            if not text_files:
                print("No text files found to process.")
                return
            
            # Process all files
            results = process_text_files(text_files, model, tokenizer, args.prompt_type, args.full_output)
            
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                # Generate output filename based on input folder
                folder_name = os.path.basename(os.path.normpath(args.folder))
                output_path = f"{folder_name}_predictions_{args.prompt_type}.txt"
            
            # Write results
            write_annotation_file(results, output_path)
            
            print("="*60)
            print(f"Batch processing completed!")
            print(f"Processed {len(results)} files")
            print(f"Results saved to: {output_path}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during batch processing: {e}")
            return

if __name__ == "__main__":
    main()
