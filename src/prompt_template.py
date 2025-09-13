from typing import List

from label_info import load_narrative_definitions, load_taxonomy, load_subnarrative_definitions


def create_category_system_prompt() -> str:
    """
    Create a system prompt for text classification that provides instructions 
    without including the text to be classified.
    """
    
    prompt = (
        "You are a strict topical classifier. Decide whether the given text is primarily about "
        "URW (Ukraine-Russia War topics) or CC (Climate Change topics).\n\n"
        "Rules (follow exactly):\n"
        "- First find the elements that indicate the topic, and reason through them step by step.\n"
        "- Output EXACTLY one label token enclosed in square brackets on the next line: [URW], [CC], or [Other].\n"
        "Classification guidance:\n"
        "- Use [URW] for topics clearly about the Russia-Ukraine conflict.\n"
        "- Use [CC] for topics clearly about climate change.\n"
        "- Use [Other] if neither topic is the primary focus.\n\n"
        "Exact example outputs (showing allowed formats):\n"
        "EVIDENCE: short justification\n[URW]\n\n"
        "EVIDENCE: short justification\n[CC]\n\n"
        "[Other]"
    )

    return prompt

def create_narrative_system_prompt(category: str, definitions_path: str = "data/narrative_definitions.csv") -> str:
    """
    Create a prompt to classify text into narratives from a specific category.
    
    Args:
        text: The input text to classify
        category: The category to filter narratives by (e.g., "URW", "CC")
        definitions_path: Path to the narrative definitions CSV file
    """
    
    # Load taxonomy and definitions
    taxonomy = load_taxonomy()
    definitions = load_narrative_definitions(definitions_path)
    
    # Get narratives for the specified category
    # The taxonomy is nested: category -> narratives -> subnarratives
    narratives = []

    if category in taxonomy:
        narratives = taxonomy[category].keys()
    
    prefixed_narratives = []
        
    # Append the category prefix to all narratives and build theme_to_narratives
    for narrative in narratives:
        prefixed_narrative = f"{category}: {narrative}"
        prefixed_narratives.append(prefixed_narrative)

    prompt_template = (
        "You are an expert propaganda narrative analyst with extensive experience in identifying and classifying manipulative communication patterns.\n"
        "Your task is to analyze the given text and identify which specific propaganda narratives are present.\n"
        "Provide your classification as a semicolon-separated list inside square brackets, e.g. [Narrative A; Narrative B].\n\n"
        "AVAILABLE NARRATIVES:\n\n"
    )

    for narrative in prefixed_narratives:
        prompt_template += f"- {narrative}\n"
        definition = definitions.get(narrative, {}).get('definition', 'No definition available.')
        example = definitions.get(narrative, {}).get('example', '')
        instruction = definitions.get(narrative, {}).get('instruction', '')
        prompt_template += f"  Definition: {definition}\n"
        if example:
            prompt_template += f"  Example: {example}\n"
        if instruction:
            prompt_template += f"  Instruction: {instruction}\n"
        prompt_template += "\n"
        
    prompt_template += f"""
    - Other : A text that does not fit any of the above narratives. Other should only be used if none of the narratives are clearly present, and should be the only narrative listed in that case.
    """
    

    prompt_template += """
## INSTRUCTIONS

Follow these two steps precisely:

**Step 1: Chain of Thought (Internal Reasoning)**
First, think step-by-step to analyze the provided text.
- Identify key phrases, arguments, and themes.
- For each potential narrative from the list, consider if it applies.
- Find a specific, direct quote from the text that serves as the strongest evidence for each narrative you believe is present.
- Formulate a brief reasoning for why that quote supports the narrative.
- If no narratives apply, conclude that.

**Step 2: Format the Final Output**
After your internal reasoning is complete, provide your final analysis as a single, valid JSON object.
- The JSON should be the *only* thing in your response. Do not include your chain of thought, explanations, or any markdown formatting like ```json.

## OUTPUT FORMAT (JSON Schema)

Your entire response must be a single JSON object conforming to this schema:
{
  "narratives": [
    {
      "narrative_name": "string (The exact name from the list)",
      "evidence_quote": "string (The direct quote from the text you found as evidence)",
      "reasoning": "string (Your justification connecting the quote to the narrative)"
    }
  ]
}

- If multiple narratives are found, include one object for each in the `narratives` list.
- If no propaganda narratives are detected, respond with an empty list for the "narratives" key: `{"narratives": []}`.
"""
    return prompt_template


def create_subnarrative_system_prompt(narrative: str, definitions_path: str = "data/subnarrative_definitions.csv") -> str:
    """
    Create a prompt to classify text into subnarratives for a specific narrative.

    Args:
        text: The input text to classify
        narrative: The parent narrative to get subnarratives for (e.g., "URW: Blaming the war on others rather than the invader")
        definitions_path: Path to the subnarrative definitions CSV file
    """
    
    # Load taxonomy and definitions
    taxonomy = load_taxonomy()
    definitions = load_subnarrative_definitions(definitions_path)
    
    # Extract category and narrative name from the input narrative
    # Expected format: "URW: Blaming the war on others rather than the invader"
    parts = narrative.split(": ", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid narrative format. Expected 'Category: Narrative Name', got: {narrative}")
    
    category, narrative_name = parts
    
    # Get subnarratives for the specified narrative from taxonomy
    subnarratives = []
    if category in taxonomy and narrative_name in taxonomy[category]:
        subnarratives = taxonomy[category][narrative_name]
    
    # Create prefixed subnarratives (with category and narrative prefix)
    prefixed_subnarratives = []
    for subnarrative in subnarratives:
        prefixed_subnarrative = f"{category}: {narrative_name}: {subnarrative}"
        prefixed_subnarratives.append(prefixed_subnarrative)

    prompt_template = (
        "You are an expert propaganda narrative analyst with extensive experience in identifying and classifying manipulative communication patterns.\n"
        "This text is known to contain the narrative: "
        f"{narrative}\n"
        "Your task is to analyze the given text and identify which specific propaganda subnarratives are present.\n"
        "Provide your classification as a semicolon-separated list inside square brackets, e.g. [Subnarrative A; Subnarrative B].\n\n"
        "AVAILABLE SUBNARRATIVES:\n\n"
    )

    for subnarrative in prefixed_subnarratives:
        prompt_template += f"- {subnarrative}\n"
        definition = definitions.get(subnarrative, {}).get('definition', 'No definition available.')
        example = definitions.get(subnarrative, {}).get('example', '')
        instruction = definitions.get(subnarrative, {}).get('instruction', '')
        prompt_template += f"  Definition: {definition}\n"
        if example:
            prompt_template += f"  Example: {example}\n"
        if instruction:
            prompt_template += f"  Instruction: {instruction}\n"
        prompt_template += "\n"

    prompt_template += """
## INSTRUCTIONS

Follow these two steps precisely:

**Step 1: Chain of Thought (Internal Reasoning)**
First, think step-by-step to analyze the provided text.
- Given that the text contains the parent narrative, look for more specific themes or arguments that match the subnarrative definitions.
- For each potential subnarrative from the list, consider if it applies.
- Find a specific, direct quote from the text that serves as the strongest evidence for each subnarrative you believe is present.
- Formulate a brief reasoning for why that quote supports the subnarrative.

**Step 2: Format the Final Output**
After your internal reasoning is complete, provide your final analysis as a single, valid JSON object.
- The JSON should be the *only* thing in your response. Do not include your chain of thought, explanations, or any markdown formatting like ```json.

## OUTPUT FORMAT (JSON Schema)

Your entire response must be a single JSON object conforming to this schema:
{
  "subnarratives": [
    {
      "subnarrative_name": "string (The exact name from the list)",
      "evidence_quote": "string (The direct quote from the text you found as evidence)",
      "reasoning": "string (Your justification connecting the quote to the subnarrative)"
    }
  ]
}

- If multiple subnarratives are found, include one object for each in the `subnarratives` list.
- If no specific subnarratives are detected, respond with an empty list for the "subnarratives" key: `{"subnarratives": []}`.
"""
    return prompt_template