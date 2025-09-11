# src/utils/model_mappings.py

"""
Model mappings for various language models and their Unsloth variants.
This module provides centralized mapping dictionaries for easy model selection.
"""

# Qwen3 Models - Standard sizes
QWEN3_MODELS = {
    "0.6b": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "1.7b": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    "4b": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    "8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "14b": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "32b": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
}

# Qwen3 MoE (Mixture of Experts) Models
QWEN3_MOE_MODELS = {
    "30b-a3b": "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
    "235b-a22b": "unsloth/Qwen3-235B-A22B-unsloth-bnb-4bit",
}

# Qwen3-2507 Latest Models (GGUF format)
QWEN3_2507_MODELS = {
    "30b-a3b-instruct": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    "30b-a3b-thinking": "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
    "235b-a22b-instruct": "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF",
    "235b-a22b-thinking": "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF",
}

# Gemma 3 Models
GEMMA3_MODELS = {
    "1b": "unsloth/Gemma-3-1B-unsloth-bnb-4bit",
    "4b": "unsloth/Gemma-3-4B-unsloth-bnb-4bit",
    "12b": "unsloth/Gemma-3-12B-unsloth-bnb-4bit",
    "27b": "unsloth/Gemma-3-27B-unsloth-bnb-4bit",
}

# Gemma 3n Models (Latest)
GEMMA3N_MODELS = {
    "e2b": "unsloth/Gemma-3n-E2B-unsloth-bnb-4bit",
    "e4b": "unsloth/Gemma-3n-E4B-unsloth-bnb-4bit",
}

# Mistral Models
MISTRAL_MODELS = {
    "small-24b-2506": "unsloth/Mistral-Small-3.2-24B-2506-unsloth-bnb-4bit",
    "small-24b-2505": "unsloth/Devstral-Small-24B-2505-unsloth-bnb-4bit",
    "nemo-12b": "unsloth/Mistral-NeMo-12B-2407-unsloth-bnb-4bit",
    "7b-v0.3": "unsloth/Mistral-7B-v0.3-unsloth-bnb-4bit",
}

def resolve_model_name(model_input, model_family="qwen3"):
    """
    Resolve model name from either size/variant or full model name.
    
    Args:
        model_input: Either a size key ('4b', '32b', etc.) or full model name
        model_family: Model family to search in ('qwen3', 'llama4', 'gemma3', etc.)
        
    Returns:
        Full model name
        
    Examples:
        resolve_model_name('4b') -> 'unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit'
        resolve_model_name('1b', 'gemma3') -> 'unsloth/Gemma-3-1B-unsloth-bnb-4bit'
        resolve_model_name('unsloth/custom-model') -> 'unsloth/custom-model'
    """
    # If it's already a full model name (contains '/'), return as-is
    if '/' in model_input:
        return model_input
    
    # Get the appropriate model mapping based on family
    family_mappings = {
        'qwen3': QWEN3_MODELS,
        'qwen3-moe': QWEN3_MOE_MODELS,
        'qwen3-2507': QWEN3_2507_MODELS,
        'gemma3': GEMMA3_MODELS,
        'gemma3n': GEMMA3N_MODELS,
        'mistral': MISTRAL_MODELS,
    }
    
    # Try the specified family first
    if model_family in family_mappings:
        models = family_mappings[model_family]
        if model_input.lower() in models:
            return models[model_input.lower()]
    
    # If not found in specified family, search through all available families
    for family, models in family_mappings.items():
        if model_input.lower() in models:
            return models[model_input.lower()]
    
    # If not found, show available options for the family
    if model_family in family_mappings:
        available_variants = list(family_mappings[model_family].keys())
        raise ValueError(f"Unknown model variant '{model_input}' for family '{model_family}'. "
                        f"Available variants: {available_variants}")
    else:
        available_families = list(family_mappings.keys())
        raise ValueError(f"Unknown model family '{model_family}'. "
                        f"Available families: {available_families}")

def get_available_models(model_family=None):
    """
    Get available models for a specific family or all families.
    
    Args:
        model_family: Optional family name to filter by
        
    Returns:
        Dictionary of available models
    """
    if model_family is None:
        return ALL_MODELS
    
    family_mappings = {
        'qwen3': QWEN3_MODELS,
        'qwen3-moe': QWEN3_MOE_MODELS,
        'qwen3-2507': QWEN3_2507_MODELS,
        'gemma3': GEMMA3_MODELS,
        'gemma3n': GEMMA3N_MODELS,
        'mistral': MISTRAL_MODELS,
    }
    
    return family_mappings.get(model_family, {})

def list_model_families():
    """
    List all available model families.
    
    Returns:
        List of model family names
    """
    return ['qwen3', 'qwen3-moe', 'qwen3-2507', 'gemma3', 'gemma3n', 'mistral']