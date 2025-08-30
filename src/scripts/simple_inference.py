#!/usr/bin/env python3
# src/scripts/simple_inference.py

import argparse
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from utils.prompt_template import create_comprehensive_prompt_template, create_simple_prompt_template

def load_model(model_path, base_model, max_seq_length=4096, chat_template="qwen3-instruct"):
    """Load the finetuned model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    # Configure chat template
    tokenizer = get_chat_template(
        tokenizer=tokenizer,
        chat_template=chat_template
    )
    
    # Load LoRA adapters if model_path is provided
    if model_path and os.path.exists(model_path):
        # Load the adapter directly from the path
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def predict_narratives(text, model, tokenizer, use_simple_prompt=True):
    """Predict narratives for the given text."""
    
    # Get prompt template
    if use_simple_prompt:
        prompt_template = create_simple_prompt_template()
    else:
        prompt_template = create_comprehensive_prompt_template()
    
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
        enable_thinking=False
    ).to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.batch_decode(outputs)
    full_response = response[0]
    
    # Extract the assistant's response
    if "<|im_start|>assistant\n" in full_response:
        assistant_response = full_response.split("<|im_start|>assistant\n")[1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.replace("<|im_end|>", "").strip()
        return assistant_response
    
    return full_response

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

def main():
    parser = argparse.ArgumentParser(description="Test inference on a single text with finetuned Qwen model.")
    
    parser.add_argument("--text", type=str, required=True,
                        help="Text to analyze for narratives.")
    parser.add_argument("--model_path", type=str, default="models/qwen/smoke-test-qwen-1.7b/final_model",
                        help="Path to finetuned model.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
                        help="Base model name.")
    parser.add_argument("--use_simple_prompt", action="store_true",
                        help="Use simple prompt template.")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                    help="The maximum sequence length of the model.")
    
    args = parser.parse_args()
    
    print("="*60)
    print("QWEN NARRATIVE DETECTION")
    print("="*60)
    print(f"Input Text: {args.text}")
    print("-"*60)
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model, max_seq_length=args.max_seq_length)
    
    # Get prediction
    raw_response = predict_narratives(args.text, model, tokenizer, args.use_simple_prompt)
    detected_narratives = parse_narratives(raw_response)
    
    # Output results
    print(f"Raw Response: {raw_response}")
    print(f"Detected Narratives: {detected_narratives}")
    print("="*60)

if __name__ == "__main__":
    main()
