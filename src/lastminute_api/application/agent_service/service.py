"""High level interface for the agent_service supervisory graph."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from functools import lru_cache
from typing import Any, Awaitable, Dict, Sequence
import logging
from typing import Dict

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


def _run_in_new_loop(coro: Awaitable[Any]) -> Any:
    """Execute ``coro`` inside a fresh event loop that closes cleanly."""

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)

        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            for task in pending:
                with suppress(asyncio.CancelledError):
                    loop.run_until_complete(task)

        loop.run_until_complete(loop.shutdown_asyncgens())
        return result
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def run_revision_agent_sync(
    query: str,
    *,
    history: Sequence[BaseMessage] | None = None,
) -> AgentState:
    """Synchronous convenience wrapper around :func:`run_revision_agent`."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – safe to spin up our own isolated loop.
        return _run_in_new_loop(run_revision_agent(query, history=history))

    raise RuntimeError(
        "run_revision_agent_sync cannot be called from an active event loop. "
        "Use 'await run_revision_agent(...)' instead."
    )


def summarise_agent_result(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a lightweight view of the agent outcome for presentation layers."""

    return {
        "answer": state.get("last_answer"),
        "image_url": state.get("image_url"),
        "query_type": state.get("query_type"),
        "mind_map_reference": state.get("mind_map_reference"),
        "messages": list(state.get("messages", [])),
    }


__all__ = [
    "run_revision_agent",
    "run_revision_agent_sync",
    "summarise_agent_result",
    "configure_agent_logging",
]
LOGGER_NAMESPACE = "lastminute_api.application.agent_service"


class EmojiFormatter(logging.Formatter):
    """Formatter that injects colour and emojis per log level."""

    LEVEL_STYLES: Dict[int, tuple[str, str]] = {
        logging.DEBUG: ("🧪", "\033[36m"),  # Cyan
        logging.INFO: ("🧭", "\033[32m"),  # Green
        logging.WARNING: ("⚠️", "\033[33m"),
        logging.ERROR: ("❌", "\033[31m"),
        logging.CRITICAL: ("🚨", "\033[35m"),
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        emoji, colour = self.LEVEL_STYLES.get(record.levelno, ("🔍", ""))
        timestamp = self.formatTime(record)
        message = record.getMessage()
        parts = [timestamp, f"{emoji} {record.levelname:<7}", record.name, message]
        line = " | ".join(parts)
        return f"{colour}{line}{self.RESET}" if colour else line

    def formatTime(self, record, datefmt=None):  # pragma: no cover - lean wrapper
        return super().formatTime(record, datefmt or "%Y-%m-%d %H:%M:%S")


def configure_agent_logging(level: int = logging.INFO) -> None:
    """Configure verbose logging for the agent service namespace."""

    logger = logging.getLogger(LOGGER_NAMESPACE)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(EmojiFormatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
