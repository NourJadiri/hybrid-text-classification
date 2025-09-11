from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import add_messages
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolCall,
    AIMessage,
)
from langgraph.func import entrypoint, task

llm = init_chat_model("openai:gpt-5-mini")

@tool
def add(a: int, b: int):
    """
    Adds two integers and returns the result.
    Args:
        a (int): The first integer to add.
        b (int): The second integer to add.
    Returns:
        int: The sum of a and b.
    """
    
    return a + b

@tool
def subtract(a: int, b: int):
    """
    Subtracts two integers.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The result of a minus b.
    """
    
    return a - b

tools = [add, subtract]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return llm_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )
    
@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)
@entrypoint()
def agent(messages: list[BaseMessage]):
    llm_response: BaseMessage = call_llm(messages).result()

    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_llm(messages).result()

    messages = add_messages(messages, llm_response)
    return messages

messages = [HumanMessage(content="Add 3 and 4.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")


