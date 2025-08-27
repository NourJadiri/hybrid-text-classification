# src/scripts/finetune_qwen.py

import argparse
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# Since this script is in src/scripts, we need to adjust the path to import from src
# This is a common pattern for making scripts runnable from the project root.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_management.loaders import load_labeled_df
from utils.prompt_template import create_comprehensive_prompt_template, create_simple_prompt_template

def format_dataset(df, tokenizer, use_comprehensive=True):
    """
    Formats the dataframe into a dataset ready for SFTTrainer.
    Each row is converted into a chat format string.
    
    Args:
        df: DataFrame with 'text' and 'narratives' columns
        tokenizer: The tokenizer to use for chat template
        use_comprehensive: Whether to use comprehensive or simple prompt template
    """
    
    # Get the appropriate prompt template
    if use_comprehensive:
        prompt_template = create_comprehensive_prompt_template()
    else:
        prompt_template = create_simple_prompt_template()
    
    # Convert the 'narratives' column (which contains lists) into a semicolon-separated string
    df['narratives_str'] = df['narratives'].apply(
        lambda x: "; ".join(x) if isinstance(x, list) and x else "Other"
    )

    # Create the full formatted text for each row
    def create_chat_prompt(row):
        # Format the user message with the text
        user_content = prompt_template.format(text=row['text'])
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"[{row['narratives_str']}]"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)

    # Apply the function to create a new column with the formatted text
    texts = df.apply(create_chat_prompt, axis=1).tolist()
    
    # Create and return a datasets.Dataset object
    return Dataset.from_dict({"text": texts})

def create_trainer(model, train_dataset, eval_dataset, args):
    
    training_args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps, # Use GA to mimic batch size!
        warmup_steps = args.warmup_steps,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        logging_steps = 5,
        optim = args.optim,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = args.seed,
        report_to = "none", # Use this for WandB etc
        load_best_model_at_end=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    return trainer


def main(args):
    """
    Main function to execute the fine-tuning pipeline.
    """
    print("--- Starting Qwen Fine-Tuning Script ---")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    
    print("\n--- Step 1: Loading Model and Tokenizer ---")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    tokenizer = get_chat_template(
        tokenizer= tokenizer,
        chat_template=args.chat_template
    )
    
    print("Model and tokenizer loaded successfully")
    
    print("\n--- Step 2: Configuring model for LoRA ---")

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed
    )
    
    print("PEFT model configured for LoRA training.")
    model.print_trainable_parameters()
    
    print("\n--- Step 3: Loading and Preparing Dataset ---")
    
    full_df = load_labeled_df(args.dataset_path)
    narratives_df = full_df[["text", "narratives"]].dropna().reset_index(drop=True)
    
    if args.num_samples:
        print(f"--- Using a subset of {args.num_samples} samples for quick testing ---")
        narratives_df = narratives_df.head(args.num_samples)
    
    print(f"Loaded {len(narratives_df)} samples from the dataset.")

    train_df, eval_df = train_test_split(narratives_df, test_size=0.1, random_state=args.seed)
    print(f"Split dataset into {len(train_df)} training samples and {len(eval_df)} evaluation samples.")
    
    train_dataset = format_dataset(train_df, tokenizer, use_comprehensive=not args.use_simple_prompt)
    eval_dataset = format_dataset(eval_df, tokenizer, use_comprehensive=not args.use_simple_prompt)
    
    print("Dataset formatted for training.")
    print("\nSample formatted training example:")
    print(train_dataset[0]['text'])
    
    print("\n--- Step 4: Creating Trainer ---")

    trainer = create_trainer(model, train_dataset, eval_dataset, args)
    
    print("Starting training...")

    trainer.train()
    
    print(f"\n--- Step 5: Saving final model and running inference test ---")
    
    # Saving the LoRA adapters
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"LoRA adapters saved to {final_model_path}")
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    # Prepare the prompt for the test text
    test_text = "The Western sanctions are only strengthening Russia's resolve and economy, while hurting European citizens. This is a strategic failure by NATO."
    
    # Use the same prompt template as training
    prompt_template = create_comprehensive_prompt_template() if not hasattr(args, 'use_simple_prompt') or not args.use_simple_prompt else create_simple_prompt_template()
    user_content = prompt_template.format(text=test_text)
    
    messages = [
        {"role": "user", "content": user_content},
    ]
    
    # Tokenize the input
    inputs = tokenizer.apply_chat_template(messages,
                                           tokenize = True,
                                           add_generation_prompt = True,
                                           return_tensors = "pt",
                                           enable_thinking=False).to("cuda")

    # Generate the response
    outputs = model.generate(input_ids = inputs)
    response = tokenizer.batch_decode(outputs)
    
    print("\n--- INFERENCE TEST ---")
    print("Input Text:", test_text)
    print("Model Response:", response[0].split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "").strip())
    print("--------------------")

    print("--- Fine-Tuning Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen model for narrative classification.")

    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", help="The model name to use from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, default="data/processed/phase0_baseline_labeled.parquet", help="Path to the labeled parquet dataset.")
    parser.add_argument("--output_dir", type=str, default="models/qwen-finetuned", help="Directory to save the fine-tuned model.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for the model.")
    parser.add_argument("--chat_template", type=str, default="qwen3-instruct", help="The chat template to use ('qwen2', 'chatml', etc.).")
    parser.add_argument("--use_simple_prompt", action="store_true", help="Use simple prompt template instead of comprehensive one.")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--max_steps", type=int, default=100, help="Total number of training steps to perform.")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use for a quick test run. Uses all data if not specified.")

    args = parser.parse_args()
    main(args)