from typing import List

from label_info import load_narrative_definitions


def create_category_prompt(text: str) -> str:
    """
    Create a concise prompt that forces the model to output a bracketed label on
    the first non-empty line: [URW], [CC], or [Other]. Optionally one EVIDENCE:
    line may follow.
    """

    prompt = (
        "You are a strict topical classifier. Decide whether the following text is primarily about "
        "URW (Ukraine-Russia War topics) or CC (Climate Change topics).\n\n"
        "Rules (follow exactly):\n"
        "- Output EXACTLY one label token enclosed in square brackets on the first non-empty line: [URW], [CC], or [Other].\n"
        "- Optionally, on the following line output 'EVIDENCE: ' followed by a short phrase (<=20 words) justifying the label.\n"
        "Classification guidance:\n"
        "- Use [URW] for topics clearly about war, conflict, sanctions, refugees, Russia/Ukraine, NATO, military operations, or geopolitical blame.\n"
        "- Use [CC] for topics clearly about climate, global warming, greenhouse gases, emissions, climate policy, renewable energy, sea level rise, or environmental impacts.\n"
        "- Use [Other] if neither topic is the primary focus.\n\n"
        "Text to classify:\n\n"
    )

    prompt += text.strip() + "\n\n"
    prompt += (
        "Produce the label now following the rules above. Exact example outputs (showing allowed formats):\n"
        "[URW]\nEVIDENCE: short justification\n\n[CC]\nEVIDENCE: short justification\n\n[Other]"
    )

    return prompt

def create_narrative_prompt_template(text: str, narratives: List[str], definitions_path: str = "data/narrative_definitions.csv") -> str:
    """
    Create a prompt to classify text into one of the provided narratives.
    The prompt lists the narratives and instructs the model to choose one.

    Args:
        text: The input text to classify
        narratives: List of narrative names to choose from
    """

    definitions = load_narrative_definitions(definitions_path)

    prompt_template = (
        "You are an expert propaganda narrative analyst with extensive experience in identifying and classifying manipulative communication patterns.\n"
        "Your task is to analyze the given text and identify which specific propaganda narratives are present.\n"
        "Provide your classification as a semicolon-separated list inside square brackets, e.g. [Narrative A; Narrative B].\n\n"
        "AVAILABLE NARRATIVES:\n\n"
    )

    for narrative in narratives:
        prompt_template += f"- {narrative}\n"
        definition = definitions[narrative]['definition'] if 'definition' in definitions[narrative] else 'No definition available.'
        example = definitions[narrative]['example'] if 'example' in definitions[narrative] else ''
        instruction = definitions[narrative]['instruction'] if 'instruction' in definitions[narrative] else ''
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
    return prompt_template