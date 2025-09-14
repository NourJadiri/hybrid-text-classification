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
- The JSON should be the *only* thing in your response. Do not include your chain of thought, explanations, or any markdown formatting like '```json'.

## OUTPUT FORMAT (JSON Schema)

Your entire response must be a single JSON object conforming to this schema:
## OUTPUT FORMAT (JSON Schema)

Your entire response MUST be a single JSON object. ALL fields in the schema below are REQUIRED. Do not omit any fields.

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
        definition = definitions.get(subnarrative, {}).get('definition', '')
        example = definitions.get(subnarrative, {}).get('example', '')
        instruction = definitions.get(subnarrative, {}).get('instruction', '')
        prompt_template += f"  Definition: {definition}\n" if definition else ""
        if example:
            prompt_template += f"  Example: {example}\n"
        if instruction:
            prompt_template += f"  Instruction: {instruction}\n"
        prompt_template += "\n"
    
    prompt_template += (
        f"- {narrative}: Other\n"
        "  Definition: Use this option if you find statements that clearly support the parent narrative but do not fit into any of the more specific subnarrative definitions provided above. This can be used IN ADDITION to other specific subnarratives.\n\n"
    )

    prompt_template += """
## INSTRUCTIONS

Follow these two steps precisely:

**Step 1: Chain of Thought (Internal Reasoning)**
First, think step-by-step to analyze the provided text.
- Given that the text contains the parent narrative, look for more specific themes or arguments that match the subnarrative definitions.
- For each potential subnarrative from the list, consider if it applies.
- Find a specific, direct quote from the text that serves as the strongest evidence for each subnarrative you believe is present.
- Formulate a brief reasoning for why that quote supports the subnarrative.

**Step 2: Check for a Remainder**
After identifying all the specific subnarratives, re-read the text. Ask yourself: "Are there any other phrases or arguments that support the parent narrative but were NOT used as evidence for the specific subnarratives I already found?"

**Step 3: Add "Other" if Necessary**
If the answer to Step 2 is YES, you MUST ALSO include the `{narrative}: Other` subnarrative in your final list.

**Step 4: Format the Final Output**
After your internal reasoning is complete, provide your final analysis as a single, valid JSON object.
- The JSON should be the *only* thing in your response. Do not include your chain of thought, explanations, or any markdown formatting like '```json'.

## EXAMPLE OF CORRECT LOGIC

- **Text says:** "Climate activists are just alarmist children. Their funding is also very suspicious."
- **Parent Narrative:** `CC: Criticism of climate movement`
- **Your Analysis:**
  - "Climate activists are just alarmist children" matches `...: Climate movement is alarmist`.
  - "Their funding is also very suspicious" supports the parent narrative but doesn't match any specific subnarrative. This is a remainder.
- **Correct Output:** A list containing BOTH `...: Climate movement is alarmist` AND `CC: Criticism of climate movement: Other`.

## Guidelines for using "...: Other"

- **DO** include `...: Other` alongside specific subnarratives if there is distinct, leftover evidence supporting the parent narrative.
- **DO NOT** include `...: Other` if all the evidence supporting the parent narrative has already been neatly captured by the specific subnarrative(s) you identified. In this case, the classification is complete without it.
- **DO** use `...: Other` as the sole entry if the text supports the parent narrative but does not align with any of the specific definitions.

## OUTPUT FORMAT (JSON Schema)

Your entire response must be a single JSON object conforming to this schema:
## OUTPUT FORMAT (JSON Schema)

Your entire response MUST be a single JSON object. ALL fields in the schema below are REQUIRED. Do not omit any fields.

{
  "narratives": [
    {
      "subnarrative_name": "string (The exact name from the list)",
      "evidence_quote": "string (The direct quote from the text you found as evidence)",
      "reasoning": "string (Your justification connecting the quote to the narrative)"
    }
  ]
}

- If multiple subnarratives are found, include one object for each in the `subnarratives` list.
- If no specific subnarratives are detected, respond with an empty list for the "subnarratives" key: `{"subnarratives": []}`.
"""
    return prompt_template


def create_narrative_critic_prompt(narratives) -> str:
    """Creates a system prompt for the 'Critic' agent that validates narrative classifications."""
    prompt = (
        "You are a meticulous and skeptical editor. Your task is to evaluate a classification of propaganda narratives applied to a text. "
        "You must be extremely strict. The classification is only valid if every narrative is strongly and explicitly supported by the provided evidence from the text.\n\n"
        "You will be given the original text and the classification analysis in JSON format. The analysis includes the narrative name, a quote for evidence, and reasoning.\n\n"
        "Here are the narratives that were available for classification:\n"
        f"{', '.join(narratives)}\n\n"
        "## EVALUATION CRITERIA (Apply Strictly):\n"
        "1. **Evidence Accuracy:** Is the `evidence_quote` an exact, verbatim quote from the original text? Of course if the quote is from the text but there was just a small formatting / typography issue it is still fine (not the same unicode character used for example).\n"
        "2. **Relevance of Evidence:** Does the `evidence_quote` DIRECTLY support the `narrative_name`? The connection must not be a stretch or require deep interpretation. If the link is weak, the classification is invalid.\n"
        "3. **Completeness:** Does the analysis miss any other obvious, high-confidence narratives that are clearly present in the text? If so, the classification is invalid.\n\n"
        "## OUTPUT FORMAT\n"
        "Provide your evaluation as a single, valid JSON object conforming to the specified schema. Do not include any other text or formatting."
    )
    return prompt

def create_subnarrative_critic_prompt(subnarratives) -> str:
    """Creates a system prompt for the 'Critic' agent that validates subnarrative classifications."""
    prompt = (
        "You are a meticulous and skeptical editor. Your task is to evaluate a classification of propaganda subnarratives applied to a text. "
        "The list was generated under the assumption that a specific parent narrative is present."
        "You must be extremely strict. The classification is only valid if every subnarrative is strongly and explicitly supported by the provided evidence from the text.\n\n"
        "You will be given the original text and the classification analysis in JSON format. The analysis includes the subnarrative name, a quote for evidence, and reasoning.\n\n"
        "Here is the list of subnarratives that were available to choose from:\n\n"
        f"{'; '.join(subnarratives)}\n\n"
        "## EVALUATION CRITERIA (Apply Strictly):\n"
        "1. **Evidence Accuracy:** Is the `evidence_quote` an exact, verbatim quote from the original text?\n"
        "2. **Relevance of Evidence:** Does the `evidence_quote` DIRECTLY and OBVIOUSLY support the `subnarrative_name`? The connection must not be a stretch or require deep interpretation. If the link is weak, the classification is invalid.\n"
        "3. **Completeness:** Does the analysis miss any other obvious, high-confidence subnarratives that are clearly present in the text? If so, the classification is invalid.\n\n"
        "## OUTPUT FORMAT\n"
        "Provide your evaluation as a single, valid JSON object conforming to the specified schema. Do not include any other text or formatting."
    )
    return prompt

def create_narrative_refinement_prompt(original_prompt: str, feedback: str) -> str:
    """Creates a system prompt for a retry, incorporating the critic's feedback."""
    refinement_header = (
        "You previously analyzed a text, but your analysis had flaws. "
        "A meticulous editor has provided the following feedback. Your task is to re-analyze the text, incorporating this feedback to produce a new, corrected classification.\n\n"
        "## EDITOR'S FEEDBACK TO CORRECT:\n"
        f"{feedback}\n\n"
        "-------------------------------------\n"
        "## ORIGINAL TASK AND DEFINITIONS (Apply these again with the feedback in mind):\n\n"
    )
    return refinement_header + original_prompt

def create_subnarrative_refinement_prompt(original_prompt: str, feedback: str) -> str:
    """Creates a system prompt for a retry of subnarrative classification, incorporating the critic's feedback."""
    refinement_header = (
        "You previously analyzed a text for subnarratives, but your analysis had flaws. "
        "A meticulous editor has provided the following feedback. Your task is to re-analyze the text, incorporating this feedback to produce a new, corrected classification.\n\n"
        "## EDITOR'S FEEDBACK TO CORRECT:\n"
        f"{feedback}\n\n"
        "-------------------------------------\n"
        "## ORIGINAL TASK AND DEFINITIONS (Apply these again with the feedback in mind):\n\n"
    )
    return refinement_header + original_prompt


def create_cleaning_system_prompt() -> str:
    """System prompt to clean raw web text by removing page noise.
    The model must return ONLY the cleaned text with no extra commentary.
    """
    return (
        "You are a precise text cleaner. Your job is to clean raw text scraped from the web by removing UI noise and boilerplate while preserving the article's content.\n\n"
        "STRICT RULES (follow exactly):\n"
        "- REMOVE: navigation menus, cookie banners, sign-up banners, button labels (e.g., 'Accept', 'Subscribe', 'Read more'), share widgets, headers/footers, unrelated CTAs, legal disclaimers, pagination artifacts, and unrelated links.\n"
        "- KEEP: the main article or post content only. Preserve sentence order, punctuation, and language.\n"
        "- DO NOT paraphrase or summarize. Do not add or remove meaning.\n"
        "- OUTPUT: Return ONLY the cleaned text with no additional commentary, headings, or JSON."
    )
