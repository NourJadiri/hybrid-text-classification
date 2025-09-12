from typing import NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from prompt_template import create_category_system_prompt

llm = init_chat_model("openai:gpt-5-mini")

class ClassificationState(TypedDict):
    """State schema for text classification workflow"""
    text: str
    category: NotRequired[str]

def classify_category_node(state: ClassificationState) -> dict:
    """
    Node function that classifies text into categories using the category prompt template.
    
    Args:
        state: Current state containing the text to classify
        
    Returns:
        Dictionary with the classification result
    """
    text = state["text"]
    
    system_prompt = create_category_system_prompt()
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]
    
    response = llm.invoke(messages)
    
    return {"category": response.content}

def create_classification_graph():
    """Create and compile the text classification graph"""
    
    builder = StateGraph(ClassificationState)
    
    builder.add_node("classify", classify_category_node)
    
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    
    graph = builder.compile()
    
    return graph

classification_graph = create_classification_graph()

with open("devset/EN/subtask-2-documents/EN_CC_200033.txt", "r", encoding="utf-8") as file:
    sample_text = file.read()
    
    sample_state = {"text": sample_text}

result = classification_graph.invoke(sample_state)
print(result)
