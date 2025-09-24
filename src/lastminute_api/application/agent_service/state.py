# src/lastminute_api/application/agent_service/state.py

from langgraph.graph import MessagesState
from typing import Optional
from typing import Literal

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
        image_url (Optional[str]): URL or path to generated image/diagram.
    """

    last_query: str
    query_type: Literal["quick_search", "complex_search"]
    continue_session: bool = False
    last_answer: Optional[str] = None
    mind_map_data: Optional[dict] = None
    image_url: Optional[str] = None
