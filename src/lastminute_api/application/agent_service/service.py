"""High level interface for the agent_service supervisory graph."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, Sequence
import logging

from langchain_core.messages import BaseMessage, HumanMessage

from lastminute_api.application.agent_service.graph import build_revision_graph
from lastminute_api.application.agent_service.state import AgentState


@lru_cache(maxsize=1)
def _get_compiled_graph():
    """Return a compiled LangGraph runnable for the revision supervisor."""
    graph = build_revision_graph()
    return graph.compile()


def _coerce_messages(history: Sequence[BaseMessage] | None) -> list[BaseMessage]:
    return list(history) if history else []


def _ensure_state(state: Mapping[str, Any] | AgentState) -> AgentState:
    """Normalise LangGraph output into an ``AgentState`` instance."""

    try:
        snapshot = dict(state)
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise TypeError("Agent state must be mapping-like") from exc

    new_state = AgentState()
    new_state.update(snapshot)
    return new_state


async def run_revision_agent(
    query: str,
    *,
    history: Sequence[BaseMessage] | None = None,
) -> AgentState:
    """Execute the full supervisor graph for a single user query."""

    if not query:
        raise ValueError("Query must be non-empty")

    messages = _coerce_messages(history)
    messages.append(HumanMessage(content=query))

    initial_state: AgentState = AgentState()
    initial_state["messages"] = messages
    initial_state["last_query"] = query
    initial_state["awaiting_subagent"] = False
    initial_state["final_response_sent"] = False

    app = _get_compiled_graph()
    result_state = await app.ainvoke(initial_state)
    return _ensure_state(result_state)


def run_revision_agent_sync(
    query: str,
    *,
    history: Sequence[BaseMessage] | None = None,
) -> AgentState:
    """Synchronous convenience wrapper around :func:`run_revision_agent`."""

    return asyncio.run(run_revision_agent(query, history=history))


def summarise_agent_result(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a lightweight view of the agent outcome for presentation layers."""

    return {
        "answer": state.get("last_answer"),
        "image_url": state.get("image_url"),
        "query_type": state.get("query_type"),
        "messages": list(state.get("messages", [])),
    }


__all__ = [
    "run_revision_agent",
    "run_revision_agent_sync",
    "summarise_agent_result",
    "configure_agent_logging",
]
LOGGER_NAMESPACE = "lastminute_api.application.agent_service"


def configure_agent_logging(level: int = logging.INFO) -> None:
    """Configure verbose logging for the agent service namespace."""

    logger = logging.getLogger(LOGGER_NAMESPACE)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
