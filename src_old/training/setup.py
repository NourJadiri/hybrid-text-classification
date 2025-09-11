import torch
from transformers.optimization import get_linear_schedule_with_warmup

def load_model_and_tokenizer(model_name, device, num_total_labels, id_to_label, label_to_id, state_dict_path=None):
    """
    Initializes or loads a model and tokenizer.

    If a state_dict_path is provided, it loads the model weights.
    Otherwise, it initializes a new model from the pre-trained model name.

    Args:
        model_name (str): The name of the pre-trained model.
        device (torch.device): The device to load the model onto.
        num_total_labels (int): Number of labels for classification.
        id_to_label (dict): Mapping from id to label.
        label_to_id (dict): Mapping from label to id.
        state_dict_path (str, optional): Path to the saved state dict. Defaults to None.

    Returns:
        model: The loaded or initialized model.
        tokenizer: The loaded tokenizer.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_total_labels,
        problem_type='multi_label_classification',
        id2label=id_to_label,
        label2id=label_to_id
    )

    if state_dict_path:
        try:
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from state dict at {state_dict_path} and tokenizer loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: State dict not found at {state_dict_path}. Initializing a new model.")
    else:
        print("Initialized new model and tokenizer successfully.")

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