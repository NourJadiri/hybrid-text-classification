"""
State definition for the LangGraph translation pipeline.
"""
from typing import TypedDict, NotRequired


class TranslationState(TypedDict):
    """State for the translation and consolidation pipeline."""
    text: str
    cleaned_text: NotRequired[str]
    translated_text: NotRequired[str]
    file_id: str
    source_language: str  # EN, BG, HI, PT, RU
