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
        "- Use [URW] for topics clearly about war, conflict, sanctions, refugees, Russia/Ukraine, NATO, military operations, or geopolitical blame.\n"
        "- Use [CC] for topics clearly about climate, global warming, greenhouse gases, emissions, climate policy, renewable energy, sea level rise, or environmental impacts.\n"
        "- Use [Other] if neither topic is the primary focus.\n\n"
        "Exact example outputs (showing allowed formats):\n"
        "EVIDENCE: short justification\n[URW]\n\n"
        "EVIDENCE: short justification\n[CC]\n\n"
        "[Other]"
    )

    return prompt

def create_narrative_prompt_template(text: str, category: str, definitions_path: str = "data/narrative_definitions.csv") -> str:
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
    

    prompt_template +="""
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
- Wrap the narrative list in square brackets ([])
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
    """
    return prompt_template.format(text=text)


def create_subnarrative_prompt_template(text: str, narrative: str, definitions_path: str = "data/subnarrative_definitions.csv") -> str:
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

    prompt_template +="""
## CLASSIFICATION INSTRUCTIONS

1. **Thorough Analysis**: Carefully read and analyze the entire text to understand its main themes, arguments, and underlying messages.
2. **Subnarrative Identification**: Identify ALL propaganda subnarratives present in the text. A single text may contain multiple subnarratives.
3. **Evidence-Based Classification**: Base your classification on the specific definitions provided above. Look for both explicit statements and implicit messaging that aligns with subnarrative patterns.
4. **Contextual Understanding**: Consider the context, tone, and intended audience when identifying subnarratives.
5. **Precision**: Only classify subnarratives that are clearly present in the text. Avoid over-interpretation or assumptions.

## OUTPUT FORMAT
Provide your analysis in the following structured format:
**IDENTIFIED SUBNARRATIVES:** [subnarrative1; subnarrative2; subnarrative3; ...]
Where:
- Wrap the subnarrative list in square brackets ([])
- Each subnarrative should be the exact name from the categories above
- Use semicolons (;) to separate multiple subnarratives
- If multiple subnarratives are present, list them all
- Maintain the exact subnarrative names as provided in the definitions

## ANALYSIS GUIDELINES
- **Primary Focus**: Identify the main propaganda subnarratives
- **Secondary Elements**: Note supporting or implicit subnarratives
- **Confidence**: Only include subnarratives you can clearly identify with evidence from the text
- **Completeness**: Ensure all significant subnarratives are captured


Please analyze the following text:
Text: {text}
    """
    return prompt_template.format(text=text)