"""Assembly utilities for the agent_service LangGraph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from lastminute_api.application.agent_service.node import (
    image_generation_node,
    mcp_agent_node,
    supervisor_node,
    tavily_agent_node,
)
from lastminute_api.application.agent_service.state import AgentState


def build_revision_graph() -> StateGraph:
    """Create and compile the supervisory LangGraph for the agent service."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tavily_agent", tavily_agent_node)
    graph.add_node("mcp_agent", mcp_agent_node)
    graph.add_node("image_agent", image_generation_node)

    graph.set_entry_point("supervisor")

    # Supervisory node uses Commands to jump dynamically, so we only need explicit
    # return edges from workers back to the supervisor.
    graph.add_edge("tavily_agent", "supervisor")
    graph.add_edge("mcp_agent", "supervisor")
    graph.add_edge("image_agent", "supervisor")

    graph.add_edge("supervisor", END)

    return graph


__all__ = ["build_revision_graph"]
