from typing import List
from schema import Narrative, Subnarrative
from state import ClassificationState


def create_other_narrative(reasoning: str = "N/A") -> Narrative:
    """
    Create an 'Other' narrative placeholder.
    
    Args:
        reasoning: Explanation for why this is classified as 'Other'
        
    Returns:
        Narrative object with 'Other' classification
    """
    return Narrative(
        narrative_name="Other",
        evidence_quote="N/A",
        reasoning=reasoning
    )


def create_other_subnarrative(reasoning: str = "N/A") -> Subnarrative:
    """
    Create an 'Other' subnarrative placeholder.
    
    Args:
        reasoning: Explanation for why this is classified as 'Other'
        
    Returns:
        Subnarrative object with 'Other' classification
    """
    return Subnarrative(
        subnarrative_name="Other",
        evidence_quote="N/A",
        reasoning=reasoning
    )


def write_classification_results(file_id: str, narratives: List[Narrative], 
                               subnarratives: List[Subnarrative], output_file: str) -> None:
    """
    Write classification results to a TSV file.
    
    Args:
        file_id: Identifier for the file being processed
        narratives: List of classified narratives
        subnarratives: List of classified subnarratives
        output_file: Path to the output file
    """
    narrative_names = [n.narrative_name for n in narratives]
    subnarrative_names = [sn.subnarrative_name for sn in subnarratives]
    
    narratives_str = ";".join(narrative_names) if narrative_names else "Other"
    subnarratives_str = ";".join(subnarrative_names) if subnarrative_names else "Other"
    
    output_line = f"{file_id}\t{narratives_str}\t{subnarratives_str}\n"
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output_line)


def route_after_category(state) -> str:
    """
    Determines the next step after category classification.
    
    Args:
        state: Current classification state
        
    Returns:
        Next node name to execute
    """
    category = state.get("category")
    
    if category == "Other":
        return "handle_other_category"
    else:
        return "narratives"

def route_after_validation(state: ClassificationState) -> str:
    """
    Determines the next step after validation.
    - If approved or max retries reached, proceed.
    - Otherwise, loop back to the actor for a retry.
    """
    feedback = state.get("narrative_validation_feedback")
    retry_count = state.get("narrative_retry_count", 0)
    
    if feedback == "approved" or retry_count >= 1: # Limit to 1 retry for now
        if retry_count >= 1 and feedback != "approved":
            print("[graph] ROUTER: Max retries reached. Proceeding with last attempt.")
        else:
            print("[graph] ROUTER: Validation approved. Proceeding to clean narratives.")
        return "clean_narratives"
    else:
        print("[graph] ROUTER: Retrying narrative classification.")
        return "narratives"
