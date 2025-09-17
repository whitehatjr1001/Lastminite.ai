from langgraph.graph import MessagesState
from typing import Optional
from typing import Literal
class AgentState(MessagesState):
    """State class for the AI Companion workflow
    Extends MessagesState to track conversation history and maintains the last message received.
    """

    summary: str
    workflow: Literal["conversation", "image"]
    image_path: Optional[str] = None
    memory_context: str