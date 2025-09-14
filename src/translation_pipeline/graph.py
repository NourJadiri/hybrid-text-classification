"""
Main LangGraph workflow for the translation pipeline.
"""
import os
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import functools

from .state import TranslationState
from .nodes import clean_text_node, translation_node, file_writing_node

load_dotenv()

# Initialize LLM (GPT-4o as requested)
llm = init_chat_model("gpt-4o", model_provider="openai")


def _clean_text_node(state: TranslationState) -> dict:
    return clean_text_node(state, llm)


def _translation_node(state: TranslationState) -> dict:
    return translation_node(state, llm)


def _file_writing_node(state: TranslationState, output_folder: str) -> dict:
    return file_writing_node(state, output_folder)


def create_translation_graph(output_folder: str):
    """Create and compile the translation graph"""
    print("[translation] Building translation graph...")
    
    # Create partial function with output folder
    write_with_folder = functools.partial(_file_writing_node, output_folder=output_folder)
    
    builder = StateGraph(TranslationState)
    
    builder.add_node("clean_text", _clean_text_node)
    builder.add_node("translate_text", _translation_node)
    builder.add_node("write_file", write_with_folder)
    
    builder.add_edge(START, "clean_text")
    builder.add_edge("clean_text", "translate_text")
    builder.add_edge("translate_text", "write_file")
    builder.add_edge("write_file", END)
    
    graph = builder.compile()
    print("[translation] Graph compiled")
    
    return graph