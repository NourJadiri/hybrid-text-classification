"""
Graph nodes for the LangGraph translation pipeline.
"""
import os
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from .state import TranslationState


def clean_text_node(state: TranslationState, llm) -> dict:
    """Clean raw web text by removing UI noise and boilerplate using an LLM."""
    raw_text = state["text"]
    print(f"[translation] Starting text cleaning for {state['file_id']}")
    
    system_prompt = """You are tasked with cleaning raw web-scraped text. Your goal is to extract the main textual content while removing:

1. Navigation menus and UI elements
2. Cookie notices and privacy policies  
3. Advertisements and promotional content
4. Footer information and contact details
5. Social media widgets and sharing buttons
6. Page headers and site branding
7. Repeated boilerplate text
8. HTML artifacts and formatting remnants

Keep:
- Main article/post content
- Headlines and subheadings that are part of the content
- Important contextual information
- Relevant quotes or citations

Return only the cleaned text without any explanations or metadata."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_text)]
    response = llm.invoke(messages)
    cleaned = (response.content or "").strip()
    
    print(f"[translation] Text cleaning complete for {state['file_id']} (length in/out: {len(raw_text)} -> {len(cleaned)})")
    return {"cleaned_text": cleaned}


def translation_node(state: TranslationState, llm) -> dict:
    """Translate text to English if needed, or pass through English text unchanged."""
    file_id = state["file_id"]
    source_language = state["source_language"]
    
    # Use cleaned text if available, otherwise use original
    text_to_translate = state.get("cleaned_text", state["text"])
    
    print(f"[translation] Processing {file_id} from language: {source_language}")
    
    if source_language == "EN":
        print(f"[translation] {file_id} is already in English, copying as-is")
        return {"translated_text": text_to_translate}
    
    print(f"[translation] Translating {file_id} from {source_language} to English")
    
    # Create translation prompt
    system_prompt = f"""You are a professional translator. Translate the following text from {get_language_name(source_language)} to English.

Requirements:
- Maintain the original meaning and tone
- Preserve the structure and formatting where possible  
- Use natural, fluent English
- Do not add explanations or commentary
- Return only the translated text

Source language: {get_language_name(source_language)}
Target language: English"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=text_to_translate)]
    response = llm.invoke(messages)
    translated_text = (response.content or "").strip()
    
    print(f"[translation] Translation complete for {file_id} (length: {len(text_to_translate)} -> {len(translated_text)})")
    return {"translated_text": translated_text}


def file_writing_node(state: TranslationState, output_folder: str) -> dict:
    """Write the translated/cleaned text to the output folder."""
    file_id = state["file_id"]
    translated_text = state["translated_text"]
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output filename (keep original file ID but ensure .txt extension)
    output_filename = f"{file_id}.txt"
    output_path = os.path.join(output_folder, output_filename)
    
    print(f"[translation] Writing {file_id} to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(translated_text)
    
    print(f"[translation] Successfully wrote {file_id}")
    return {}

def get_language_name(lang_code: str) -> str:
    """Convert language code to full name."""
    lang_map = {
        "EN": "English",
        "BG": "Bulgarian", 
        "HI": "Hindi",
        "PT": "Portuguese",
        "RU": "Russian"
    }
    return lang_map.get(lang_code, lang_code)


def determine_language_and_translation_need(file_path: str) -> tuple[str, bool]:
    """
    Determine source language and whether translation is needed based on file path.
    Returns (language_code, needs_translation)
    """
    path_parts = Path(file_path).parts
    
    # Find language folder in path
    for part in path_parts:
        if part in ["EN", "BG", "HI", "PT", "RU"]:
            language = part
            needs_translation = (language != "EN")
            return language, needs_translation
    
    # Default fallback
    return "EN", False