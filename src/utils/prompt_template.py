"""
Comprehensive prompt template functions for narrative classification with Qwen
"""

import pandas as pd
import os
from typing import Dict, List, Optional


def load_narrative_definitions(file_path: str = "data/narrative_definitions.csv") -> Dict[str, Dict[str, str]]:
    """
    Load narrative definitions from CSV file.
    
    Args:
        file_path: Path to the narrative definitions CSV file
        
    Returns:
        Dictionary mapping narrative names to their definitions and examples
    """
    try:
        df = pd.read_csv(file_path)
        definitions = {}
        
        for _, row in df.iterrows():
            narrative = row['narrative']
            definition = row['definition']
            example = row.get('example', '') if pd.notna(row.get('example', '')) else ''
            instruction = row.get('instruction for annotator', '') if pd.notna(row.get('instruction for annotator', '')) else ''
            
            definitions[narrative] = {
                'definition': definition,
                'example': example,
                'instruction': instruction
            }
        
        return definitions
    
    except FileNotFoundError:
        print(f"Warning: Narrative definitions file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading narrative definitions: {e}")
        return {}


def get_unique_narratives_from_definitions(definitions_path: str = "data/narrative_definitions.csv") -> List[str]:
    """
    Extract unique narrative names from the definitions file.
    
    Args:
        definitions_path: Path to the narrative definitions CSV file
        
    Returns:
        List of unique narrative names
    """
    try:
        df = pd.read_csv(definitions_path)
        # Remove duplicates while preserving order
        narratives = df['narrative'].drop_duplicates().tolist()
        return narratives
    except Exception as e:
        print(f"Error extracting narratives: {e}")
        return []


def create_comprehensive_prompt_template(
    definitions_path: str = "data/narrative_definitions.csv"
) -> str:
    """
    Create a comprehensive prompt template for narrative classification.
    
    Args:
        definitions_path: Path to the narrative definitions CSV file
        
    Returns:
        Complete prompt template string
    """
    
    # Load narrative definitions
    definitions = load_narrative_definitions(definitions_path)
    narratives = get_unique_narratives_from_definitions(definitions_path)
    
    # Classify narratives by category
    urw_narratives = [n for n in narratives if n.startswith('URW:')]
    cc_narratives = [n for n in narratives if n.startswith('CC:')]
    other_narratives = [n for n in narratives if not n.startswith(('URW:', 'CC:'))]
    
    # Build the prompt template
    prompt_template = """You are an expert propaganda narrative analyst with extensive experience in identifying and classifying manipulative communication patterns. Your expertise encompasses political messaging, media analysis, and the detection of subtle propaganda techniques across different domains.

Your task is to analyze the given text and identify which specific propaganda narratives are present. You must provide your classification as a structured output in the form of an array where elements are separated by semicolons.

## AVAILABLE NARRATIVE CATEGORIES

### URW (Ukraine, Russia, War) Related Narratives:
"""
    
    # Add URW narrative definitions
    for narrative in urw_narratives:
        if narrative in definitions:
            prompt_template += f"\n**{narrative}**\n"
            prompt_template += f"Definition: {definitions[narrative]['definition']}\n"
            if definitions[narrative]['example']:
                prompt_template += f"Examples: {definitions[narrative]['example']}\n"
    
    prompt_template += "\n### CC (Climate Change) Related Narratives:\n"
    
    # Add CC narrative definitions
    for narrative in cc_narratives:
        if narrative in definitions:
            prompt_template += f"\n**{narrative}**\n"
            prompt_template += f"Definition: {definitions[narrative]['definition']}\n"
            if definitions[narrative]['example']:
                prompt_template += f"Examples: {definitions[narrative]['example']}\n"
    
    # Add other categories if they exist
    if other_narratives:
        prompt_template += "\n### Other Categories:\n"
        for narrative in other_narratives:
            if narrative in definitions:
                prompt_template += f"\n**{narrative}**\n"
                prompt_template += f"Definition: {definitions[narrative]['definition']}\n"
                if definitions[narrative]['example']:
                    prompt_template += f"Examples: {definitions[narrative]['example']}\n"
    
    # Add classification instructions
    prompt_template += """
## CLASSIFICATION INSTRUCTIONS

1. **Thorough Analysis**: Carefully read and analyze the entire text to understand its main themes, arguments, and underlying messages.

2. **Narrative Identification**: Identify ALL propaganda narratives present in the text. A single text may contain multiple narratives.

3. **Evidence-Based Classification**: Base your classification on the specific definitions provided above. Look for both explicit statements and implicit messaging that aligns with narrative patterns.

4. **Contextual Understanding**: Consider the context, tone, and intended audience when identifying narratives.

5. **Precision**: Only classify narratives that are clearly present in the text. Avoid over-interpretation or assumptions.

## OUTPUT FORMAT

Provide your analysis in the following structured format:

**IDENTIFIED NARRATIVES:** [narrative1; narrative2; narrative3; ...]

Where:
- Each narrative should be the exact name from the categories above
- Use semicolons (;) to separate multiple narratives
- If no propaganda narratives are detected, respond with: [Other]
- If multiple narratives are present, list them all
- Maintain the exact narrative names as provided in the definitions

## ANALYSIS GUIDELINES

- **Primary Focus**: Identify the main propaganda narratives
- **Secondary Elements**: Note supporting or implicit narratives
- **Confidence**: Only include narratives you can clearly identify with evidence from the text
- **Completeness**: Ensure all significant narratives are captured

Please analyze the following text:

Text: {text}

**IDENTIFIED NARRATIVES:**"""
    
    return prompt_template


def create_simple_prompt_template() -> str:
    """
    Create a simpler prompt template for basic classification.
    
    Returns:
        Simple prompt template string
    """
    
    prompt_template = """You are an expert at analyzing text for propaganda narratives. Your task is to classify the given text and identify which narratives are present.

Available narrative categories include:
- URW (Ukraine, Russia, War related): Various subcategories like "Discrediting Ukraine", "Praise of Russia", "Discrediting the West", etc.
- CC (Climate Change): Various subcategories like "Amplifying Climate Fears", "Criticism of climate policies", etc.
- Other: For texts that don't fit the main categories

Please analyze the following text and identify all relevant narratives present.

IMPORTANT: Return your response as semicolon-separated narrative labels in the format:
[narrative1; narrative2; narrative3]

Text: {text}

IDENTIFIED NARRATIVES:"""

    return prompt_template

def create_constrained_prompt_template(
    definitions_path: str = "data/narrative_definitions.csv"
) -> str:
    """
    Creates a prompt with the core instruction and a list of all possible narrative labels.
    """
    # We can reuse this function to get the list of labels
    narratives = get_unique_narratives_from_definitions(definitions_path)
    
    # Format the list of narratives for the prompt
    narrative_list_str = "\n- ".join(narratives)
    
    prompt_template = f"""You are a propaganda analyst. Your task is to identify which of the following propaganda narratives are present in the given text.

## AVAILABLE NARRATIVES
- {narrative_list_str}
- Other

## INSTRUCTIONS
- Analyze the text and identify all matching narratives from the list above.
- Your response MUST be a semicolon-separated list of the exact narrative names inside brackets.
- For example: [narrative1; narrative2]
- If no narratives from the list are found, respond with [Other].

Text: {{text}}

IDENTIFIED NARRATIVES:"""
    
    return prompt_template

def format_prompt_for_training(
    text: str, 
    narratives: List[str], 
    use_comprehensive: bool = True,
    definitions_path: str = "data/narrative_definitions.csv"
) -> str:
    """
    Format a complete training example with text and expected narratives.
    
    Args:
        text: Input text to classify
        narratives: List of expected narrative labels
        use_comprehensive: Whether to use comprehensive or simple template
        definitions_path: Path to narrative definitions file
        
    Returns:
        Formatted training prompt with expected output
    """
    
    if use_comprehensive:
        template = create_comprehensive_prompt_template(definitions_path)
    else:
        template = create_simple_prompt_template()
    
    # Format the prompt with the text
    formatted_prompt = template.format(text=text)
    
    # Add the expected response
    if isinstance(narratives, list):
        response = "; ".join(narratives) if narratives else "Other"
    else:
        response = str(narratives)
    
    return f"{formatted_prompt} [{response}]"


def format_chat_for_training(
    text: str, 
    narratives: List[str], 
    use_comprehensive: bool = True,
    definitions_path: str = "data/narrative_definitions.csv"
) -> Dict[str, List[Dict[str, str]]]:
    """
    Format a training example in chat format for fine-tuning.
    
    Args:
        text: Input text to classify
        narratives: List of expected narrative labels
        use_comprehensive: Whether to use comprehensive or simple template
        definitions_path: Path to narrative definitions file
        
    Returns:
        Chat format dictionary with messages
    """
    
    if use_comprehensive:
        template = create_comprehensive_prompt_template(definitions_path)
    else:
        template = create_simple_prompt_template()
    
    # Format the user message
    user_message = template.format(text=text)
    
    # Format the expected response
    if isinstance(narratives, list):
        response = "; ".join(narratives) if narratives else "Other"
    else:
        response = str(narratives)
    
    # Return in chat format
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"[{response}]"}
        ]
    }


def test_prompt_templates():
    """
    Test function to demonstrate prompt template usage.
    """
    
    # Test data
    test_text = "Les pays de l'europe sont entrain d'appliquer des sanctions injustes sur le gouvernement Russe. Cela profite clairement a l'ukraine qui continue de s'enrichir grace a son programme de genocide nazi. La russie est une nation forte et resiliente qui ne merite pas ce traitement"
    test_narratives = ["URW: Russia is the Victim", "URW: Discrediting Ukraine", "URW: Praise of Russia"]
    
    print("=== COMPREHENSIVE PROMPT TEMPLATE ===")
    comprehensive_prompt = create_comprehensive_prompt_template()
    print(comprehensive_prompt.format(text=test_text))
    
    print("\n" + "="*60 + "\n")
    
    print("=== SIMPLE PROMPT TEMPLATE ===")
    simple_prompt = create_simple_prompt_template()
    print(simple_prompt.format(text=test_text))
    
    print("\n" + "="*60 + "\n")
    
    print("=== CHAT FORMAT EXAMPLE ===")
    chat_format = format_chat_for_training(test_text, test_narratives, use_comprehensive=False)
    print("User:", chat_format['messages'][0]['content'][:200] + "...")
    print("Assistant:", chat_format['messages'][1]['content'])


if __name__ == "__main__":
    test_prompt_templates()
