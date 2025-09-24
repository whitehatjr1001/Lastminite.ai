# src/lastminute_api/application/agent_service/nodes.py

from typing import Annotated, Literal
from langchain.tools import tool
from langgraph.types import Command
from lastminute_api.application.agent_service.state import AgentState as State


# === Custom Handoff Tool ===

@tool
def custom_handoff_tool(
    agent_name: Annotated[str, "Name of the sub-agent to hand off to (e.g., 'tavily', 'mcp', 'image')"],
    task_input: Annotated[str, "Input or query string for the sub-agent"],
):
    """
    Custom handoff tool signaling supervisor to delegate to sub-agent with input.
    Returns a Command instructing the graph to go to the named agent node with state update.
    """
    return Command(
        goto=agent_name,
        update={"last_query": task_input},
        graph=Command.PARENT  # Ensures navigation within parent graph
    )


# === Supervisor Node ===

def supervisor_node(state: State) -> Command[Literal["tavily_agent", "mcp_agent", "image_agent", "__end__"]]:
    """
    Supervisor node decides which sub-agent to call next and calls the handoff tool.

    Uses a simple heuristic to choose agent and performs handoff via custom tool.
    """
    query = state.last_query if hasattr(state, "last_query") else ""

    # Simple routing logic demo (customize as required)
    if not query:
        return Command(goto="__end__")

    if len(query) < 20:
        # Hand off to Tavily quick search agent
        return custom_handoff_tool("tavily_agent", query)

    # After quick search, delegate complex reasoning
    if getattr(state, "current_task", None) == "quick_search":
        return custom_handoff_tool("mcp_agent", query)

    if getattr(state, "current_task", None) == "complex_search":
        return custom_handoff_tool("image_agent", query)

    # Start with Tavily agent by default
    return custom_handoff_tool("tavily_agent", query)


# === Stub Sub-Agent Nodes ===

def tavily_agent_node(state: State) -> Command[Literal["supervisor"]]:
    # Simulate quick search and return control to supervisor
    answer = f"Tavily quick search result for: {state.last_query}"
    updated_state = state.copy()
    updated_state.last_answer = answer
    updated_state.chat_response = answer
    updated_state.current_task = "quick_search"
    return Command(goto="supervisor", update=updated_state.dict())

def mcp_agent_node(state: State) -> Command[Literal["supervisor"]]:
    # Simulate complex MCP reasoning and return to supervisor
    answer = f"MCP reasoning result for: {state.last_query}"
    updated_state = state.copy()
    updated_state.last_answer = answer
    updated_state.chat_response = answer
    updated_state.current_task = "complex_search"
    return Command(goto="supervisor", update=updated_state.dict())

def image_generation_node(state: State) -> Command[Literal["supervisor"]]:
    # Simulate image generation and return control
    image_url = "https://dummy.image/mindmap.png"
    updated_state = state.copy()
    updated_state.image_url = image_url
    updated_state.current_task = "image_generation"
    return Command(goto="supervisor", update=updated_state.dict())
