from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, conlist
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    timeout=None,
    max_retries=2,
)

class Article(BaseModel):
    # We add a 'persona' field to see what style the LLM used
    writing_style: str = Field(description="The persona or style used for this article, e.g., 'Skeptical Economist', 'Optimistic Technologist'.")
    generated_text: str = Field(description="The complete, multi-paragraph journalistic text written in the specified style.")
    
    
class GeneratedArticleBatch(BaseModel):
    """A Pydantic model for a batch of generated articles."""
    # conlist ensures we get a list of at least 8 distinct articles
    articles: conlist(Article, min_length=8) = Field(description="A list of generated articles, each from a different perspective.")
    

def create_narrative_generator_chain():
    """
    Creates a LangChain generator chain using the provided LLM.
    
    Args:
        llm: The language model to use for text generation.
        
    Returns:
        A LangChain chain configured for text generation.
    """
    print("Creating generator chain...")
    
    prompt_template_str = """
    You are an expert journalist and linguist generating diverse, high-quality training data.
    Your objective is to write multiple, distinct news-style texts that are all clear examples of the provided narrative.

    **Narrative to Generate:**
    {narrative_name}

    **Narrative Definition:**
    {narrative_def}
    
    **Example:**
    {narrative_example}

    **Your Task:**
    Generate a batch of **{num_examples} distinct and diverse texts** that all exemplify the narrative above. To ensure diversity, write each text from a **different persona or perspective**.

    Here are some example personas to adopt for each text:
    - A skeptical financial journalist focusing on economic angles.
    - An optimistic technologist writing for a tech publication.
    - A concerned environmental scientist from a research institute.
    - A conservative political commentator.
    - A liberal policy advisor.
    - A local news reporter focusing on community impact.
    - An international relations expert.
    - A historian providing context.

    **Crucial Constraints for EACH text:**
    1.  **Style:** Each text must be in a formal, journalistic style appropriate to its persona.
    2.  **Structure:** Each text must be at least two paragraphs long. Each paragraph should be at least 5 sentences long.
    3.  **Precision:** Each text MUST clearly exemplify the target narrative.
    4.  **Factual Integrity:** DO NOT invent specific, verifiable details (names, dates, stats), if possible use them actually.

    **Format Instructions:**
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template_str)

    structured_llm = llm.with_structured_output(GeneratedArticleBatch)

    chain = prompt | structured_llm

    print("Generator chain created successfully.")
    
    return chain

def create_subnarrative_generator_chain():
    """
    Creates a LangChain generator chain specifically for sub-narratives.
    
    Args:
        llm: The language model to use for text generation.
        
    Returns:
        A LangChain chain configured for sub-narrative text generation.
    """
    print("Creating sub-narrative generator chain...")
    
    prompt_template_str = """
    You are an expert journalist and linguist generating diverse, high-quality training data.
    Your objective is to write multiple, distinct news-style texts that are all clear examples of the provided narrative.

    **Narrative to Generate:**
    {subnarrative_name}

    **Narrative Definition:**
    {subnarrative_def}
    
    **Example:**
    {subnarrative_example}

    **Your Task:**
    Generate a batch of **{num_examples} distinct and diverse texts** that all exemplify the narrative above. To ensure diversity, write each text from a **different persona or perspective**.

    
    Here are some example personas to adopt for each text:
    - A skeptical financial journalist focusing on economic angles.
    - An optimistic technologist writing for a tech publication.
    - A concerned environmental scientist from a research institute.
    - A conservative political commentator.
    - A liberal policy advisor.
    - A local news reporter focusing on community impact.
    - An international relations expert.
    - A historian providing context.

    **Crucial Constraints for EACH text:**
    1.  **Style:** Each text must be in a formal, journalistic style appropriate to its persona.
    2.  **Structure:** Each text must be at least two paragraphs long. each paragraph should be at least 5 sentences long.
    3.  **Precision:** Each text MUST clearly exemplify the target narrative.
    4.  **Factual Integrity:** DO NOT invent specific, verifiable details (names, dates, stats), if possible use them actually.

    **Format Instructions:**
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template=prompt_template_str)

    structured_llm = llm.with_structured_output(GeneratedArticleBatch)

    chain = prompt | structured_llm

    print("Sub-narrative generator chain created successfully.")
    
    return chain