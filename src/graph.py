import os
from typing import NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from extract import extract_category, extract_narratives, extract_subnarratives
from label_info import flatten_taxonomy, load_taxonomy
from prompt_template import create_category_system_prompt, create_narrative_system_prompt, create_subnarrative_system_prompt, create_subnarrative_system_prompt
from utils import get_texts_in_folder

llm = init_chat_model("openai:gpt-5-mini")
taxonomy = load_taxonomy()
flat_narratives, flat_subnarratives = flatten_taxonomy(taxonomy)

class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    category: NotRequired[str]
    narratives: NotRequired[list[str]]
    subnarratives: NotRequired[list[str]]
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

def classify_narratives_node(state: ClassificationState) -> dict:
    text = state["text"]
    category = state["category"]

    print(f"[graph] Starting narratives classification node for category {category}")

    system_prompt = create_narrative_system_prompt(category)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]

    response = llm.invoke(messages)
    narratives = extract_narratives(response.content)

    def _has_category_prefix(n: str, cat: str) -> bool:
        s = n.lstrip()
        return s.lower().startswith(f"{cat.lower()}:")
    
    narratives = [
        n if _has_category_prefix(n, category) or n == "Other" else f"{category}: {n}"
        for n in (narratives or [])
    ]
    print(f"[graph] Narratives classification complete -> {narratives}")

    return {"narratives": narratives}

def clean_narratives_node(state: ClassificationState) -> dict:
    narratives = state["narratives"]
    
    print(f"[graph] Cleaning narratives")
    cleaned_narratives = [n for n in narratives if n in flat_narratives]
    
    print(f"[graph] Cleaned narratives -> {cleaned_narratives}")

    return {"narratives": cleaned_narratives}

async def classify_subnarratives_node(state: ClassificationState) -> dict:
    text = state["text"]
    narratives = state["narratives"]

    print(f"[graph] Starting subnarratives classification node for narratives {narratives}")

    all_messages = []

    for narrative in narratives:
        system_prompt = create_subnarrative_system_prompt(narrative)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        all_messages.append(messages)

    print("[graph] Starting the batch LLM invocation for subnarratives...")
    responses = await llm.abatch(all_messages)
    print("[graph] Completed the batch LLM invocation for subnarratives.")

    all_subnarratives = []

    for response in responses:
        subnarratives = extract_subnarratives(response.content)
        all_subnarratives.extend(subnarratives or [])

    print(f"[graph] Subnarratives classification complete -> {all_subnarratives}")

    return {"subnarratives": all_subnarratives}

def clean_subnarratives_node(state: ClassificationState) -> dict:
    subnarratives = state["subnarratives"]
    
    print(f"[graph] Cleaning subnarratives")
    cleaned_subnarratives = [sn for sn in subnarratives if sn in flat_subnarratives]
    
    print(f"[graph] Cleaned subnarratives -> {cleaned_subnarratives}")

    return {"subnarratives": cleaned_subnarratives}

def write_results_node(state: ClassificationState) -> dict:
    file_id = state["file_id"]
    
    narratives = state.get("narratives", ['Other'])
    subnarratives = state.get("subnarratives", ['Other'])
    print(f"[graph] Writing results for file {file_id}")
    
    narratives_str = ";".join(narratives) if narratives else "Other"
    subnarratives_str = ";".join(subnarratives) if subnarratives else "Other"

    output_line = f"{file_id}\t{narratives_str}\t{subnarratives_str}\n"

    try:
        with open("results/langgraph_results.txt", "a", encoding="utf-8") as f:
            f.write(output_line)
        print(f"[graph] Results written to results/langgraph_results.txt")
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
    builder.add_node("write_results", write_results_node)

    builder.add_edge(START, "categories")
    builder.add_edge("categories", "narratives")
    builder.add_edge("narratives", "clean_narratives")
    builder.add_edge("clean_narratives", "subnarratives")
    builder.add_edge("subnarratives", "clean_subnarratives")
    builder.add_edge("clean_subnarratives", "write_results")
    builder.add_edge("write_results", END)
    
    graph = builder.compile()
    print("[graph] Graph compiled")

    return graph

classification_graph = create_classification_graph()
config = {
    "max_concurrency": 10
}

text_list, file_names = get_texts_in_folder("devset/EN/subtask-2-documents/")

initial_states_batch = [{"text": text, "file_id": file_id} for text, file_id in zip(text_list[:20], file_names[:20])]
print(f"[graph] Initial states batch prepared with {len(initial_states_batch)} items.")

async def main():
    results = await classification_graph.abatch(initial_states_batch, config=config)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
