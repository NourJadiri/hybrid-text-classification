import os
from typing import Any, List, NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from extract import extract_category, extract_narratives, extract_subnarratives
from label_info import flatten_taxonomy, load_taxonomy
from prompt_template import create_category_system_prompt, create_narrative_system_prompt, create_subnarrative_system_prompt, create_subnarrative_system_prompt
from schema import Narrative, NarrativeClassificationOutput, Subnarrative, SubnarrativeClassificationOutput
from utils import get_texts_in_folder
from dotenv import load_dotenv
import sys

load_dotenv()

MODEL = "openai:gpt-5-mini"
llm = init_chat_model(MODEL)
OUTPUT_FILE = f"results/langgraph_results_{MODEL}_structured_output.txt"

taxonomy = load_taxonomy()
flat_narratives, flat_subnarratives = flatten_taxonomy(taxonomy)

class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    category: NotRequired[str]
    narratives: NotRequired[List[Any]]
    subnarratives: NotRequired[list[Any]]
    file_id: str

def classify_category_node(state: ClassificationState) -> dict:
    """
    Node function that classifies text into categories using the category prompt template.
    
    Args:
        state: Current state containing the text to classify
        
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

def handle_other_category_node(state: ClassificationState) -> dict:
    """
    Handles the case where the category is 'Other'.
    It sets narratives and subnarratives to ['Other'] to bypass the main pipeline.
    """
    print("[graph] Category is 'Other', taking shortcut.")
    other_narrative = Narrative(narrative_name="Other", evidence_quote="N/A", reasoning="Category is Other")
    other_subnarrative = Subnarrative(subnarrative_name="Other", evidence_quote="N/A", reasoning="Category is Other")
    return {
        "narratives": [other_narrative],
        "subnarratives": [other_subnarrative]
    }
    
def route_after_category(state: ClassificationState) -> str:
    """
    Determines the next step after category classification.
    - If category is 'Other', routes to the shortcut node.
    - Otherwise, proceeds with narrative classification.
    """
    category = state.get("category")

    if category == "Other":
        # Return the name of the shortcut node
        return "handle_other_category"
    else:
        # Return the name of the normal next node
        return "narratives"

def classify_narratives_node(state: ClassificationState) -> dict:
    text = state["text"]
    category = state["category"]

    print(f"[graph] Starting narratives classification node for category {category}")

    system_prompt = create_narrative_system_prompt(category)
    structured_llm = llm.with_structured_output(NarrativeClassificationOutput)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]

    response_obj = structured_llm.invoke(messages)
    
    narratives_with_details = response_obj.narratives
    narrative_names = [n.narrative_name for n in narratives_with_details]

    print(f"[graph] Narratives classification complete -> {narrative_names}")

    return {"narratives": narratives_with_details}

def clean_narratives_node(state: ClassificationState) -> dict:
    narratives = state["narratives"]
    
    print(f"[graph] Cleaning narratives")
    cleaned_narratives = [narrative for narrative in narratives if narrative.narrative_name in flat_narratives]

    return {"narratives": cleaned_narratives}

async def classify_subnarratives_node(state: ClassificationState) -> dict:
    text = state["text"]
    narratives = state["narratives"]
    
    if not narratives:
        print("[graph] No valid narratives to process for subnarratives. Skipping.")
        
        other_placeholder = Subnarrative(
            subnarrative_name="Other",
            evidence_quote="N/A",
            reasoning="No valid parent narratives were provided to this node."
        )
        return {"subnarratives": [other_placeholder]}
    
    
    print(f"[graph] Starting subnarratives classification node for narratives {narratives}")
    
    structured_llm = llm.with_structured_output(SubnarrativeClassificationOutput)
    
    all_messages = []

    for narrative in narratives:
        system_prompt = create_subnarrative_system_prompt(narrative.narrative_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        all_messages.append(messages)

    print("[graph] Starting the batch LLM invocation for subnarratives...")
    responses = await structured_llm.abatch(all_messages)
    print("[graph] Completed the batch LLM invocation for subnarratives.")

    all_subnarratives_with_details = []

    for parent_narrative, response in zip(narratives, responses):
        if response.subnarratives:
            all_subnarratives_with_details.extend(response.subnarratives)
        else:
            # If the LLM returned an empty list, create a specific "Other" placeholder
            print(f"[graph] No specific subnarratives found for '{parent_narrative.narrative_name}'. Creating 'Other' placeholder.")
            other_placeholder = Subnarrative(
                subnarrative_name=f"{parent_narrative.narrative_name}: Other",
                evidence_quote="N/A",
                reasoning=f"No specific subnarrative found for the parent narrative."
            )
            all_subnarratives_with_details.append(other_placeholder)

    return {"subnarratives": all_subnarratives_with_details}


def clean_subnarratives_node(state: ClassificationState) -> dict:
    subnarratives = state["subnarratives"]
    narratives = state["narratives"]
    
    print(f"[graph] Cleaning subnarratives")
    cleaned_subnarratives = [subnarrative for subnarrative in subnarratives if subnarrative.subnarrative_name in flat_subnarratives]
    
    return {"subnarratives": cleaned_subnarratives}

def write_results_node(state: ClassificationState) -> dict:
    file_id = state["file_id"]

    narratives = state["narratives"]
    subnarratives = state["subnarratives"]

    narrative_names = [n.narrative_name for n in narratives]
    subnarrative_names = [sn.subnarrative_name for sn in subnarratives]
    print(f"[graph] Writing results for file {file_id}")

    narratives_str = ";".join(narrative_names) if narrative_names else "Other"
    subnarratives_str = ";".join(subnarrative_names) if subnarrative_names else "Other"

    output_line = f"{file_id}\t{narratives_str}\t{subnarratives_str}\n"

    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(output_line)
        print(f"[graph] Results written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"[graph] Error writing results: {e}")

    return None


def create_classification_graph():
    """Create and compile the text classification graph"""
    print("[graph] Building classification graph...")

    builder = StateGraph(ClassificationState)
    
    builder.add_node("categories", classify_category_node)
    builder.add_node("narratives", classify_narratives_node)
    builder.add_node("clean_narratives", clean_narratives_node)
    builder.add_node("subnarratives", classify_subnarratives_node)
    builder.add_node("clean_subnarratives", clean_subnarratives_node)
    builder.add_node("handle_other_category", handle_other_category_node)
    builder.add_node("write_results", write_results_node)

    builder.add_edge(START, "categories")
    builder.add_conditional_edges("categories", route_after_category)
    builder.add_edge("narratives", "clean_narratives")
    builder.add_edge("clean_narratives", "subnarratives")
    builder.add_edge("subnarratives", "clean_subnarratives")

    builder.add_edge("handle_other_category", "write_results")
    builder.add_edge("clean_subnarratives", "write_results")
    
    builder.add_edge("write_results", END)
    
    graph = builder.compile()
    print("[graph] Graph compiled")

    return graph

classification_graph = create_classification_graph()
config = {
    "max_concurrency": 20
}

text_list, file_names = get_texts_in_folder("testset/EN/subtask-2-documents/")

initial_states_batch = [{"text": text, "file_id": file_id} for text, file_id in zip(text_list, file_names)]
print(f"[graph] Initial states batch prepared with {len(initial_states_batch)} items.")

async def main():
    await classification_graph.abatch(initial_states_batch, config=config)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
