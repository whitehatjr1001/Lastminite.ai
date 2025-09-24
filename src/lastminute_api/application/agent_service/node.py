# src/lastminute_api/application/agent_service/nodes.py

from typing import Literal
from langgraph.types import Command
from lastminute_api.application.agent_service.state import AgentState as State

# === Supervisor Node ===

def supervisor_node(state: State) -> Command[Literal["tavily_agent", "mcp_agent", "image_agent", "supervisor", "__end__"]]:
    """
    Supervisor node that decides which sub-agent node to run next.

    The decision may be based on:
    - The last user query complexity (simple or complex)
    - Whether image generation is needed
    - How many cycles have run (to avoid infinite loops)
    - Any other state flags

    Returns:
        Command object indicating next node to go to with optional state updates.
    """
    query = state.last_query if hasattr(state, "last_query") else ""

    # Example simple heuristic routing:
    if not query:
        # No query, end the workflow
        return Command(goto="__end__")

    if len(query) < 20:
        # Simple short query -> Tavily quick search
        return Command(goto="tavily_agent", update={"current_task": "quick_search"})

    # For demonstration, after MCP run, run image generation
    if getattr(state, "current_task", None) == "quick_search":
        # After tavily search, run MCP for deeper reasoning
        return Command(goto="mcp_agent", update={"current_task": "complex_search"})

    if getattr(state, "current_task", None) == "complex_search":
        # After MCP, run image generation
        return Command(goto="image_agent", update={"current_task": "image_generation"})

    # After image generation, end flow
    if getattr(state, "current_task", None) == "image_generation":
        return Command(goto="__end__")

    # Default fallback: start with Tavily
    return Command(goto="tavily_agent", update={"current_task": "quick_search"})

# === Tavily Agent Node ===

def tavily_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Node implementing quick fact search logic, simulating a Tavily quick search.

    Fetches fast answers for simple queries.

    Returns:
        Command to return control to supervisor with updated state.
    """
    # TODO: Integrate your actual Tavily quick search here
    answer = f"Quick Tavily search answer for query: '{state.last_query}'"

    updated_state = state.copy()
    updated_state.last_answer = answer
    updated_state.chat_response = answer

    return Command(goto="supervisor", update=updated_state.dict())

# === MCP Agent Node ===

def mcp_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Node implementing complex MCP multi-tool reasoning logic.

    Handles in-depth queries using multi-server MCP agents.

    Returns:
        Command to return control to supervisor with updated state.
    """
    # TODO: Place your MCP multi-agent tool logic here
    answer = f"Detailed MCP reasoning answer for query: '{state.last_query}'"

    updated_state = state.copy()
    updated_state.last_answer = answer
    updated_state.chat_response = answer

    return Command(goto="supervisor", update=updated_state.dict())

# === Image Generation Node ===

def image_generation_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Node for generating mind map or visualization images from the query or answers.

    Returns:
        Command to return control to supervisor with updated state including image URL.
    """
    # TODO: Integrate your nano_bannana or other image gen here
    image_url = "https://dummy.image/mindmap.png"

    updated_state = state.copy()
    updated_state.image_url = image_url

    return Command(goto="supervisor", update=updated_state.dict())
