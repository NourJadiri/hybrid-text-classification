"""
Node functions for the LangGraph classification pipeline.
Extracted from graph.py for better code organization and readability.
"""

import json
from typing import Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from extract import extract_category
from label_info import flatten_taxonomy
from prompt_template import create_category_system_prompt, create_narrative_critic_prompt, create_narrative_refinement_prompt, create_narrative_system_prompt, create_subnarrative_system_prompt
from schema import Narrative, NarrativeClassificationOutput, Subnarrative, SubnarrativeClassificationOutput, ValidationResult
from graph_utils import create_other_narrative, create_other_subnarrative, write_classification_results


def classify_category_node(state, llm) -> dict:
    """
    Node function that classifies text into categories using the category prompt template.
    
    Args:
        state: Current state containing the text to classify
        llm: The language model instance
        
    Returns:
        Dictionary with the classification result
    """
    text = state["text"]
    print("[graph] Starting category classification node")
    
    system_prompt = create_category_system_prompt()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]
    
    response = llm.invoke(messages)
    category = extract_category(response.content)
    
    print(f"[graph] Category classification complete -> {category}")
    return {"category": category}


def handle_other_category_node(state) -> dict:
    """
    Handles the case where the category is 'Other'.
    Sets narratives and subnarratives to ['Other'] to bypass the main pipeline.
    """
    print("[graph] Category is 'Other', taking shortcut.")
    
    other_narrative = create_other_narrative("Category is Other")
    other_subnarrative = create_other_subnarrative("Category is Other")
    
    return {
        "narratives": [other_narrative],
        "subnarratives": [other_subnarrative]
    }


def classify_narratives_node(state, llm) -> dict:
    """
    Node function that classifies text into narratives for a given category.
    
    Args:
        state: Current state containing text and category
        llm: The language model instance
        
    Returns:
        Dictionary with the narrative classification results
    """
    text = state["text"]
    category = state["category"]
    retry_count = state.get("narrative_retry_count", 0)
    feedback = state.get("narrative_validation_feedback", "")
    
    print(f"[graph] Starting narratives classification node for category {category}")
    
    base_system_prompt = create_narrative_system_prompt(category)
    
    if feedback and feedback != "approved":
        print("[graph] ACTOR: This is a retry. Incorporating feedback.")
        system_prompt = create_narrative_refinement_prompt(base_system_prompt, feedback)
    else:
        system_prompt = base_system_prompt
        
        
    structured_llm = llm.with_structured_output(NarrativeClassificationOutput)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]
    
    response_obj = structured_llm.invoke(messages)
    narratives_with_details = response_obj.narratives
    narrative_names = [n.narrative_name for n in narratives_with_details]
    
    print(f"[graph] Narratives classification complete -> {narrative_names}")
    return {"narratives": narratives_with_details, "narrative_retry_count": retry_count + 1}


def validate_narratives_node(state, llm) -> dict:
    """
    The 'Critic' node. It validates the narratives classified by the actor.
    """
    print("[graph] CRITIC: Validating classified narratives...")
    text = state["text"]
    narratives_analysis = state["narratives"] 
    
    if not narratives_analysis:
        print("[graph] CRITIC: No narratives to validate. Approving.")
        return {"narrative_validation_feedback": "approved"}

    system_prompt = create_narrative_critic_prompt()
    
    human_prompt = (
        f"ORIGINAL TEXT:\n---\n{text}\n---\n\n"
        f"CLASSIFICATION ANALYSIS TO VALIDATE:\n---\n{json.dumps([n.dict() for n in narratives_analysis], indent=2)}\n---"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    critic_llm = llm.with_structured_output(ValidationResult)
    verdict = critic_llm.invoke(messages)

    if verdict.is_valid:
        print(f"[graph] CRITIC: Validation successful. Feedback: {verdict.feedback}")
        return {"narrative_validation_feedback": "approved"}
    else:
        print(f"[graph] CRITIC: Validation failed. Feedback: {verdict.feedback}")
        return {"narrative_validation_feedback": verdict.feedback}


def clean_narratives_node(state, flat_narratives) -> dict:
    """
    Node function that filters narratives to only include valid ones.
    
    Args:
        state: Current state containing narratives to clean
        flat_narratives: List of valid narrative names
        
    Returns:
        Dictionary with cleaned narratives
    """
    narratives = state["narratives"]
    
    if not narratives:
        print("[graph] No narratives to clean. Skipping.")
        other_placeholder = create_other_narrative("No valid narratives were provided to this node.")
        return {"narratives": [other_placeholder]}
    
    print("[graph] Cleaning narratives")
    cleaned_narratives = [
        narrative for narrative in narratives 
        if narrative.narrative_name in flat_narratives
    ]
    
    return {"narratives": cleaned_narratives}


async def classify_subnarratives_node(state, llm) -> dict:
    """
    Node function that classifies text into subnarratives for given narratives.
    Uses batch processing for efficiency.
    
    Args:
        state: Current state containing text and narratives
        llm: The language model instance
        
    Returns:
        Dictionary with the subnarrative classification results
    """
    text = state["text"]
    narratives = state["narratives"]
    
    if not narratives:
        print("[graph] No valid narratives to process for subnarratives. Skipping.")
        other_placeholder = create_other_subnarrative("No valid parent narratives were provided to this node.")
        return {"subnarratives": [other_placeholder]}
    
    print(f"[graph] Starting subnarratives classification node for narratives {narratives}")
    
    # Prepare batch messages
    structured_llm = llm.with_structured_output(SubnarrativeClassificationOutput)
    all_messages = _prepare_subnarrative_messages(narratives, text)
    
    # Process batch
    print("[graph] Starting the batch LLM invocation for subnarratives...")
    responses = await structured_llm.abatch(all_messages)
    print("[graph] Completed the batch LLM invocation for subnarratives.")
    
    # Process responses
    all_subnarratives_with_details = _process_subnarrative_responses(narratives, responses)
    
    return {"subnarratives": all_subnarratives_with_details}


def clean_subnarratives_node(state, flat_subnarratives) -> dict:
    """
    Node function that filters subnarratives to only include valid ones.
    
    Args:
        state: Current state containing subnarratives to clean
        flat_subnarratives: List of valid subnarrative names
        
    Returns:
        Dictionary with cleaned subnarratives
    """
    subnarratives = state["subnarratives"]
    
    print("[graph] Cleaning subnarratives")
    cleaned_subnarratives = [
        subnarrative for subnarrative in subnarratives 
        if subnarrative.subnarrative_name in flat_subnarratives
    ]
    
    return {"subnarratives": cleaned_subnarratives}


def write_results_node(state, output_file) -> dict|None:
    """
    Node function that writes classification results to file.
    
    Args:
        state: Current state containing results to write
        output_file: Path to the output file
        
    Returns:
        None (end node)
    """
    file_id = state["file_id"]
    narratives = state["narratives"]
    subnarratives = state["subnarratives"]
    
    print(f"[graph] Writing results for file {file_id}")
    
    # Add "Other" placeholders if needed
    final_subnarratives = _add_other_placeholders_if_needed(narratives, subnarratives)
    
    try:
        write_classification_results(file_id, narratives, final_subnarratives, output_file)
        print(f"[graph] Results written to {output_file}")
    except Exception as e:
        print(f"[graph] Error writing results: {e}")
    
    return None


# Private helper functions

def _add_other_placeholders_if_needed(narratives, subnarratives):
    """
    Add "Other" placeholders for narratives that don't have any subnarratives.
    
    Args:
        narratives: List of narrative objects
        subnarratives: List of subnarrative objects
        
    Returns:
        Updated list of subnarratives with "Other" placeholders added as needed
    """
    final_subnarratives = list(subnarratives)  # Copy the original list
    
    for narrative in narratives:
        # Check if this narrative has any subnarratives
        has_subnarratives = any(
            sub.subnarrative_name.startswith(narrative.narrative_name)
            for sub in subnarratives
        )
        
        if not has_subnarratives:
            print(f"[graph] No specific subnarratives found for '{narrative.narrative_name}'. Creating 'Other' placeholder.")
            other_placeholder = Subnarrative(
                subnarrative_name=f"{narrative.narrative_name}: Other",
                evidence_quote="N/A",
                reasoning="No specific subnarrative found for the parent narrative."
            )
            final_subnarratives.append(other_placeholder)
    
    return final_subnarratives


def _prepare_subnarrative_messages(narratives, text):
    """Prepare batch messages for subnarrative classification."""
    all_messages = []
    for narrative in narratives:
        system_prompt = create_subnarrative_system_prompt(narrative.narrative_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        all_messages.append(messages)
    return all_messages


def _process_subnarrative_responses(narratives, responses):
    """Process batch responses from subnarrative classification."""
    all_subnarratives_with_details = []
    
    for parent_narrative, response in zip(narratives, responses):
        if response.subnarratives:
            all_subnarratives_with_details.extend(response.subnarratives)
        # Placeholder logic moved to write_results_node via _add_other_placeholders_if_needed
    
    return all_subnarratives_with_details

