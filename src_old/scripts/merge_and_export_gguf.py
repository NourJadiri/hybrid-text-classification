# src/scripts/merge_and_export_gguf.py

import argparse
import os
from unsloth import FastLanguageModel

def main(args):
    print("--- Starting LoRA Merge and GGUF Export ---")

    if not os.path.exists(args.adapter_path):
        print(f"ERROR: Adapter path not found at '{args.adapter_path}'")
        return

    print(f"Loading model from adapter path: {args.adapter_path}")
    
    # Load the fine-tuned model from the adapter path.
    # Unsloth automatically finds the base model and merges the adapters.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    output_filename = f"{args.output_name}.gguf"
    print(f"Exporting merged model to GGUF format: {output_filename}")
    print(f"Using quantization method: {args.quantization_method}")

    # Save the merged model in GGUF format
    model.save_pretrained_gguf(
        output_filename,
        tokenizer,
        quantization_method=args.quantization_method
    )

    print("\n" + "="*50)
    print(" GGUF Export Complete!")
    print(f"Your model is saved as: {output_filename}")
    print("You can now use this file with Ollama.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and export to GGUF for Ollama.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the fine-tuned LoRA adapter directory (e.g., 'models/qwen-1.7b-cot-finetune/final_model').")
    parser.add_argument("--output_name", type=str, required=True, help="Base name for the output GGUF file (e.g., 'qwen-1.7b-narrative-detector').")
    parser.add_argument("--quantization_method", type=str, default="q4_k_m", help="The quantization method to use (e.g., q4_k_m, q8_0, f16).")
    
    args = parser.parse_args()
    main(args)