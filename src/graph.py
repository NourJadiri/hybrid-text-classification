from typing import NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from extract import extract_category, extract_narratives, extract_subnarratives
from prompt_template import create_category_system_prompt, create_narrative_system_prompt, create_subnarrative_system_prompt, create_subnarrative_system_prompt

llm = init_chat_model("openai:gpt-5-mini")

class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    category: NotRequired[str]
    narratives: NotRequired[list[str]]
    subnarratives: NotRequired[list[str]]

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
        n if _has_category_prefix(n, category) else f"{category}: {n}"
        for n in (narratives or [])
    ]
    print(f"[graph] Narratives classification complete -> {narratives}")

    return {"narratives": narratives}

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

def create_classification_graph():
    """Create and compile the text classification graph"""
    print("[graph] Building classification graph...")

    builder = StateGraph(ClassificationState)
    
    builder.add_node("categories", classify_category_node)
    builder.add_node("narratives", classify_narratives_node)
    builder.add_node("subnarratives", classify_subnarratives_node)

    builder.add_edge(START, "categories")
    builder.add_edge("categories", "narratives")
    builder.add_edge("narratives", "subnarratives")
    builder.add_edge("subnarratives", END)

    graph = builder.compile()
    print("[graph] Graph compiled")

    return graph

classification_graph = create_classification_graph()

with open("devset/EN/subtask-2-documents/EN_CC_200033.txt", "r", encoding="utf-8") as file:
    sample_text = file.read()
    
    sample_state = {"text": sample_text}


async def main():
    result = await classification_graph.ainvoke(sample_state)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    