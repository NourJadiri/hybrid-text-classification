"""
Node functions for the LangGraph classification pipeline.
Extracted from graph.py for better code organization and readability.
"""

import json
from typing import Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from extract import extract_category
from prompt_template import create_category_system_prompt, create_narrative_critic_prompt, create_narrative_refinement_prompt, create_narrative_system_prompt, create_subnarrative_critic_prompt, create_subnarrative_refinement_prompt, create_subnarrative_system_prompt, create_cleaning_system_prompt
from schema import NarrativeClassificationOutput, Subnarrative, SubnarrativeClassificationOutput, ValidationResult
from graph_utils import create_other_narrative, create_other_subnarrative, write_classification_results
from utils import get_narratives_for_category, get_subnarratives_for_narrative


def clean_text_node(state, llm) -> dict:
    """Clean raw web text by removing UI noise and boilerplate using an LLM."""
    raw_text = state["text"]
    print("[graph] Starting text cleaning node")
    system_prompt = create_cleaning_system_prompt()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_text)]
    response = llm.invoke(messages)
    cleaned = (response.content or "").strip()
    print(f"[graph] Text cleaning complete (length in/out: {len(raw_text)} -> {len(cleaned)})")
    return {"cleaned_text": cleaned}


def classify_category_node(state, llm) -> dict:
    """
    Node function that classifies text into categories using the category prompt template.
    
    Args:
        state: Current state containing the text to classify
        llm: The language model instance
        
    Returns:
        Dictionary with the classification result
    """
    text = state.get("cleaned_text", state["text"])  # use cleaned text if available
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
    text = state.get("cleaned_text", state["text"])  # use cleaned text if available
    category = state["category"]
    retry_count = state.get("narrative_retry_count", 0)
    feedback = state.get("narrative_validation_feedback", "")
    
    print(f"[graph] Starting narratives classification node for category {category}")
    
    base_system_prompt = create_narrative_system_prompt(category)
    
    if feedback and feedback != "approved":
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

def handle_empty_narratives_node(state) -> dict:
    """
    Handles the case where no narratives were classified.
    Sets subnarratives to empty list for proper handling in write_results.
    """
    print("[graph] No narratives classified, setting empty subnarratives.")
    
    return {
        "subnarratives": []
    }

def validate_narratives_node(state, llm, taxonomy) -> dict:
    """
    The 'Critic' node. It validates the narratives classified by the actor.
    """
    print("[graph] CRITIC: Validating classified narratives...")
    text = state.get("cleaned_text", state["text"])  # use cleaned text if available
    narratives_analysis = state["narratives"]
    category = state["category"]

    potential_narratives = get_narratives_for_category(taxonomy, category)

    if not narratives_analysis:
        print("[graph] CRITIC: No narratives to validate. Approving.")
        return {"narrative_validation_feedback": "approved"}

    system_prompt = create_narrative_critic_prompt(potential_narratives)

    human_prompt = (
        f"ORIGINAL TEXT:\n---\n{text}\n---\n\n"
        f"CLASSIFICATION ANALYSIS TO VALIDATE:\n---\n{json.dumps([n.dict() for n in narratives_analysis], indent=2)}\n---"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    critic_llm = llm.with_structured_output(ValidationResult)
    verdict = critic_llm.invoke(messages)

    if verdict.is_valid:
        print(f"[graph] CRITIC: Validation successful.")
        return {"narrative_validation_feedback": "approved"}
    else:
        print(f"[graph] CRITIC: Validation failed.")
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
    text = state.get("cleaned_text", state["text"])  # use cleaned text if available
    narratives = state["narratives"]
    retry_count = state.get("subnarrative_retry_count", 0)
    feedback = state.get("subnarrative_validation_feedback", "")
    
    valid_parent_narratives = [
        n for n in narratives if n.narrative_name != "Other"
    ]
    
    narrative_names = [n.narrative_name for n in valid_parent_narratives]
    print(f"[graph] Starting subnarratives classification node for narratives {narrative_names}")


    structured_llm = llm.with_structured_output(SubnarrativeClassificationOutput)
    all_messages = _prepare_subnarrative_messages(narratives, text, feedback)
    
    print("[graph] Starting the batch LLM invocation for subnarratives...")
    responses = await structured_llm.abatch(all_messages)
    print("[graph] Completed the batch LLM invocation for subnarratives.")
    
    all_subnarratives_with_details = _process_subnarrative_responses(narratives, responses)
    
    return {"subnarratives": all_subnarratives_with_details, "subnarrative_retry_count": retry_count + 1}

def validate_subnarratives_node(state, llm, taxonomy) -> dict:
    """
    The 'Critic' node for subnarratives. It validates the subnarratives classified by the actor.
    """
    print("[graph] CRITIC: Validating classified subnarratives...")
    text = state.get("cleaned_text", state["text"])  # use cleaned text if available
    subnarratives_analysis = state["subnarratives"]
    narratives_analysis = state["narratives"]
    category = state["category"]

    narrative_names = [n.narrative_name.split(": ")[-1] for n in narratives_analysis if n.narrative_name != "Other"]

    potential_subnarratives = []
    
    for narrative in narrative_names:
        subs = get_subnarratives_for_narrative(taxonomy, category, narrative)
        potential_subnarratives.extend(subs)
    
    if not subnarratives_analysis:
        print("[graph] CRITIC: No subnarratives to validate. Approving.")
        return {"subnarrative_validation_feedback": "approved"}

    system_prompt = create_subnarrative_critic_prompt(potential_subnarratives)
    
    human_prompt = (
        f"ORIGINAL TEXT:\n---\n{text}\n---\n\n"
        f"PARENT NARRATIVES:\n---\n{json.dumps([n.dict() for n in state['narratives']], indent=2)}\n---\n\n"
        f"CLASSIFICATION ANALYSIS TO VALIDATE:\n---\n{json.dumps([s.dict() for s in subnarratives_analysis], indent=2)}\n---"
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    critic_llm = llm.with_structured_output(ValidationResult)
    verdict = critic_llm.invoke(messages)

    if verdict.is_valid:
        print(f"[graph] CRITIC: Validation successful.")
        return {"subnarrative_validation_feedback": "approved"}
    else:
        print(f"[graph] CRITIC: Validation failed.")
        return {"subnarrative_validation_feedback": verdict.feedback}


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
    narratives = state.get("narratives", [])
    subnarratives = state.get("subnarratives", [])
    
    print(f"[graph] Writing results for file {file_id}")
    
    try:
        write_classification_results(file_id, narratives, subnarratives, output_file)
        print(f"[graph] Results written to {output_file}")
    except Exception as e:
        print(f"[graph] Error writing results: {e}")
    
    return None


async def multi_agent_classify_narratives_node(state, llm, num_agents: int) -> dict:
    """
    Node function that runs n agents in parallel to classify narratives.
    
    Args:
        state: Current state containing text and category
        llm: The language model instance
        num_agents: Number of agents to run in parallel
        
    Returns:
        Dictionary with multi-agent narrative classification results
    """
    text = state.get("cleaned_text", state["text"])
    category = state["category"]
    
    print(f"[graph] Starting multi-agent narratives classification with {num_agents} agents for category {category}")
    
    # Prepare the same prompt for all agents
    system_prompt = create_narrative_system_prompt(category)
    structured_llm = llm.with_structured_output(NarrativeClassificationOutput)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]
    
    # Run n agents in parallel
    all_messages = [messages] * num_agents
    responses = await structured_llm.abatch(all_messages)
    
    # Extract narratives from each agent
    multi_agent_results = []
    for i, response_obj in enumerate(responses):
        narratives_with_details = response_obj.narratives if response_obj.narratives else []
        narrative_names = [n.narrative_name for n in narratives_with_details]
        print(f"[graph] Agent {i+1} classified narratives: {narrative_names}")
        multi_agent_results.append(narratives_with_details)
    
    print(f"[graph] Multi-agent narratives classification complete")
    return {"multi_agent_narratives": multi_agent_results}


async def multi_agent_classify_subnarratives_node(state, llm, num_agents: int) -> dict:
    """
    Node function that runs n agents in parallel to classify subnarratives.
    
    Args:
        state: Current state containing text and narratives
        llm: The language model instance  
        num_agents: Number of agents to run in parallel
        
    Returns:
        Dictionary with multi-agent subnarrative classification results
    """
    text = state.get("cleaned_text", state["text"])
    narratives = state["narratives"]
    
    valid_parent_narratives = [
        n for n in narratives if n.narrative_name != "Other"
    ]
    
    narrative_names = [n.narrative_name for n in valid_parent_narratives]
    print(f"[graph] Starting multi-agent subnarratives classification with {num_agents} agents for narratives {narrative_names}")
    
    if not valid_parent_narratives:
        print(f"[graph] No valid narratives for subnarrative classification")
        return {"multi_agent_subnarratives": []}
    
    structured_llm = llm.with_structured_output(SubnarrativeClassificationOutput)
    
    # Collect all subnarratives from all agents across all narratives
    all_subnarratives_from_all_agents = []
    
    for narrative in valid_parent_narratives:
        system_prompt = create_subnarrative_system_prompt(narrative.narrative_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        
        # Run n agents for this narrative
        all_messages = [messages] * num_agents
        responses = await structured_llm.abatch(all_messages)
        
        # Collect results from all agents for this narrative
        for i, response in enumerate(responses):
            subnarratives = response.subnarratives if response.subnarratives else []
            subnarrative_names = [s.subnarrative_name for s in subnarratives]
            print(f"[graph] Agent {i+1} for narrative '{narrative.narrative_name}' classified subnarratives: {subnarrative_names}")
            all_subnarratives_from_all_agents.extend(subnarratives)
    
    print(f"[graph] Multi-agent subnarratives classification complete")
    # Store as nested structure: [all_subnarratives] - single list for easier aggregation
    return {"multi_agent_subnarratives": [all_subnarratives_from_all_agents]}


def aggregate_multi_agent_narratives(state, method: str) -> dict:
    """
    Aggregates multi-agent narrative classification results using intersection or union.
    
    Args:
        state: Current state containing multi-agent narrative results
        method: "intersection" or "union"
        
    Returns:
        Dictionary with aggregated narratives
    """
    print(f"[graph] Aggregating multi-agent narrative results using {method} method")
    
    # Aggregate narratives
    multi_agent_narratives = state.get("multi_agent_narratives", [])
    if multi_agent_narratives:
        aggregated_narratives = _aggregate_narratives(multi_agent_narratives, method)
    else:
        aggregated_narratives = state.get("narratives", [])
    
    narrative_names = [n.narrative_name for n in aggregated_narratives]
    print(f"[graph] Aggregated narratives: {narrative_names}")
    
    return {
        "narratives": aggregated_narratives
    }


def aggregate_multi_agent_subnarratives(state, method: str) -> dict:
    """
    Aggregates multi-agent subnarrative classification results using intersection or union.
    
    Args:
        state: Current state containing multi-agent subnarrative results
        method: "intersection" or "union"
        
    Returns:
        Dictionary with aggregated subnarratives
    """
    print(f"[graph] Aggregating multi-agent subnarrative results using {method} method")
    
    # Aggregate subnarratives
    multi_agent_subnarratives = state.get("multi_agent_subnarratives", [])
    if multi_agent_subnarratives:
        aggregated_subnarratives = _aggregate_subnarratives(multi_agent_subnarratives, method)
    else:
        aggregated_subnarratives = state.get("subnarratives", [])
    
    subnarrative_names = [s.subnarrative_name for s in aggregated_subnarratives]
    print(f"[graph] Aggregated subnarratives: {subnarrative_names}")
    
    return {
        "subnarratives": aggregated_subnarratives
    }


def _aggregate_narratives(multi_agent_results, method: str):
    """
    Aggregate narrative results from multiple agents.
    """
    if not multi_agent_results:
        return []
    
    if method == "intersection":
        # Find narratives that appear in ALL agent results
        if not multi_agent_results:
            return []
        
        # Get narrative names from each agent
        agent_narrative_sets = []
        for agent_results in multi_agent_results:
            narrative_names = {n.narrative_name for n in agent_results}
            agent_narrative_sets.append(narrative_names)
        
        # Find intersection
        common_narratives = set.intersection(*agent_narrative_sets) if agent_narrative_sets else set()
        
        # Return narrative objects for common narratives (from first agent)
        result = []
        for narrative in multi_agent_results[0]:
            if narrative.narrative_name in common_narratives:
                result.append(narrative)
        return result
        
    else:  # union
        # Find narratives that appear in ANY agent result
        seen_narratives = {}
        for agent_results in multi_agent_results:
            for narrative in agent_results:
                if narrative.narrative_name not in seen_narratives:
                    seen_narratives[narrative.narrative_name] = narrative
        
        return list(seen_narratives.values())


def _aggregate_subnarratives(multi_agent_results, method: str):
    """
    Aggregate subnarrative results from multiple agents.
    Note: multi_agent_results is now a list containing a single list of all subnarratives.
    """
    if not multi_agent_results or not multi_agent_results[0]:
        return []
    
    # Get the flat list of all subnarratives from all agents
    all_subnarratives = multi_agent_results[0]
    
    if method == "intersection":
        # Find subnarratives that appear multiple times (indicating agreement between agents)
        subnarrative_counts = {}
        subnarrative_objects = {}
        
        for subnarrative in all_subnarratives:
            name = subnarrative.subnarrative_name
            subnarrative_counts[name] = subnarrative_counts.get(name, 0) + 1
            subnarrative_objects[name] = subnarrative
        
        # For intersection, we want subnarratives that appeared multiple times
        result = []
        for name, count in subnarrative_counts.items():
            if count > 1:  # Appeared in multiple agent results
                result.append(subnarrative_objects[name])
        return result
        
    else:  # union
        # Return all unique subnarratives
        seen_subnarratives = {}
        for subnarrative in all_subnarratives:
            if subnarrative.subnarrative_name not in seen_subnarratives:
                seen_subnarratives[subnarrative.subnarrative_name] = subnarrative
        
        return list(seen_subnarratives.values())


# Private helper functions

def _prepare_subnarrative_messages(narratives, text, feedback):
    """Prepare batch messages for subnarrative classification."""
    all_messages = []
    for narrative in narratives:
        base_system_prompt = create_subnarrative_system_prompt(narrative.narrative_name)
        
        if feedback and feedback != "approved":
            print("[graph] ACTOR: This is a retry for subnarratives. Incorporating feedback.")
            system_prompt = create_subnarrative_refinement_prompt(base_system_prompt, feedback)
        else:
            system_prompt = base_system_prompt
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

