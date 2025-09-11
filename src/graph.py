from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "gpt-5-mini")

