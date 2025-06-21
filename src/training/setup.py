
import torch
from transformers.optimization import get_linear_schedule_with_warmup

def init_model_and_tokenizer(model_name, device, num_total_labels, id_to_label, label_to_id):
    """
    Initializes the model and tokenizer for the given model name.
    
    Args:
        model_name (str): The name of the pre-trained model.
        device (torch.device): The device to load the model onto (CPU or GPU).
        
    Returns:
        model: The initialized model.
        tokenizer: The initialized tokenizer.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_total_labels,
        problem_type = 'multi_label_classification',
        id2label = id_to_label,
        label2id = label_to_id 
    )

    print("Model and tokenizer loaded successfully.")
    
    # Move the model to the specified device
    model.to(device)
    
    return model, tokenizer


def setup_optimizer_and_scheduler(model, train_dataloader, epochs, learning_rate=2e-5):
    from torch.optim import AdamW
    
    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print("Optimizer set up successfully.")

    num_training_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # 10% of training steps as warmup
        num_training_steps=num_training_steps
    )

    print("Scheduler set up successfully.")

    return optimizer, scheduler