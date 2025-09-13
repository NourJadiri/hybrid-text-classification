import os
from typing import Any, List, NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from label_info import flatten_taxonomy, load_taxonomy
from state import ClassificationState
from utils import get_texts_in_folder
from graph_nodes import (
    classify_category_node,
    handle_other_category_node,
    handle_empty_narratives_node,
    classify_narratives_node,
    clean_narratives_node,
    classify_subnarratives_node,
    clean_subnarratives_node,
    validate_narratives_node,
    write_results_node
)
from graph_utils import route_after_category, route_after_narrative_validation, route_after_narratives_classification, route_after_narratives_cleaning
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL = "openai:gpt-5-nano"
llm = init_chat_model(MODEL)
OUTPUT_FILE = f"results/{MODEL.split(':')[1]}/no_subnarrative_validation.txt"

# Load taxonomy data
taxonomy = load_taxonomy()
flat_narratives, flat_subnarratives = flatten_taxonomy(taxonomy)


# Wrapper functions to inject dependencies into node functions
def _classify_category_node(state: ClassificationState) -> dict:
    return classify_category_node(state, llm)


def _handle_other_category_node(state: ClassificationState) -> dict:
    return handle_other_category_node(state)


def _handle_empty_narratives_node(state: ClassificationState) -> dict:
    return handle_empty_narratives_node(state)


def _classify_narratives_node(state: ClassificationState) -> dict:
    return classify_narratives_node(state, llm)

def _validate_narratives_node(state: ClassificationState) -> dict:
    return validate_narratives_node(state, llm, taxonomy)


def _clean_narratives_node(state: ClassificationState) -> dict:
    return clean_narratives_node(state, flat_narratives)


async def _classify_subnarratives_node(state: ClassificationState) -> dict:
    return await classify_subnarratives_node(state, llm)

def _clean_subnarratives_node(state: ClassificationState) -> dict:
    return clean_subnarratives_node(state, flat_subnarratives)


def _write_results_node(state: ClassificationState) -> dict|None:
    return write_results_node(state, OUTPUT_FILE)


def create_classification_graph():
    """Create and compile the text classification graph"""
    print("[graph] Building classification graph...")

    builder = StateGraph(ClassificationState)
    
    # Add nodes using wrapper functions
    builder.add_node("categories", _classify_category_node)
    builder.add_node("narratives", _classify_narratives_node)
    builder.add_node("validate_narratives", _validate_narratives_node)
    builder.add_node("clean_narratives", _clean_narratives_node)
    builder.add_node("subnarratives", _classify_subnarratives_node)
    builder.add_node("clean_subnarratives", _clean_subnarratives_node)
    builder.add_node("handle_other_category", _handle_other_category_node)
    builder.add_node("handle_empty_narratives", _handle_empty_narratives_node)
    builder.add_node("write_results", _write_results_node)

    # Define graph flow
    builder.add_edge(START, "categories")
    builder.add_conditional_edges("categories", route_after_category)
    
    builder.add_conditional_edges("narratives", route_after_narratives_classification)
    builder.add_conditional_edges("validate_narratives", route_after_narrative_validation)

    builder.add_conditional_edges("clean_narratives", route_after_narratives_cleaning)
    builder.add_edge("subnarratives", "clean_subnarratives")
    
    builder.add_edge("handle_other_category", "write_results")
    builder.add_edge("handle_empty_narratives", "write_results")
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

initial_states_batch = [{"text": text, "file_id": file_id} for text, file_id in zip(text_list[:5], file_names[:5])]
print(f"[graph] Initial states batch prepared with {len(initial_states_batch)} items.")

async def main():
    await classification_graph.abatch(initial_states_batch, config=config)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
