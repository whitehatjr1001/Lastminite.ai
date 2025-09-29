# src/lastminute_api/application/agent_service/state.py

from langgraph.graph import MessagesState
from typing import Optional, Literal

class AgentState(MessagesState):
    """
    State class for the LangGraph MCP Revision Agent.

    Extends MessagesState to track conversation history and session-specific data.

    Attributes:
        last_query (str): The most recent user query.
        query_type (Literal): Classifies query as 'quick_search' or 'complex_search'.
        continue_session (bool): Whether user wants to continue revision session.
        last_answer (Optional[str]): Last generated textual answer.
        mind_map_data (Optional[dict]): Serialized graph data for mind map visualization.
        mind_map_context (Optional[str]): Consolidated study notes used to build the mind map.
        mind_map_url (Optional[str]): Data URL of the rendered mind map image.
        mind_map_summary (Optional[str]): Narrative explanation of the mind map.
        image_url (Optional[str]): URL or path to generated image/diagram (includes mind maps).
    """

    last_query: str = ""
    query_type: Optional[Literal["simple_answer", "quick_search", "deep_research", "image_generation", "mind_map"]] = None
    continue_session: bool = False
    last_answer: Optional[str] = None
    current_task: Optional[str] = None
    mind_map_data: Optional[dict] = None
    mind_map_context: Optional[str] = None
    mind_map_url: Optional[str] = None
    mind_map_summary: Optional[str] = None
    image_url: Optional[str] = None
    chat_response: Optional[str] = None
    awaiting_subagent: bool = False
    final_response_sent: bool = False
    mcp_query: Optional[str] = None
