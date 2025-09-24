"""Supervisory nodes for the agent_service layer.

This module wires a supervisor that can either respond directly or
delegate to specialised workers (quick web search, deep MCP research,
image generation). A custom hand-off tool is exposed so the supervisor
can switch execution to the appropriate node while carrying state
updates across the LangGraph.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Annotated, Dict, Literal, Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

from lastminute_api.application.agent_service.prompts import get_prompt
from lastminute_api.application.agent_service.state import AgentState as State
from lastminute_api.infrastructure.llm_providers.base import get_llm_by_type
from lastminute_api.infrastructure.mcp.mcp_agent import create_agent as create_mcp_agent
from lastminute_api.infrastructure.nano_bannana.openai import create_openai_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_llm():  # pragma: no cover - thin wrapper for clarity
    return get_llm_by_type("openai")


def _extract_user_query(state: State) -> str:
    """Return the latest user utterance and persist it on the state."""
    query = state.get("last_query") or ""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        # HumanMessage, dict, or tuple depending on caller
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        elif isinstance(last_message, dict):
            if last_message.get("role") == "user":
                query = last_message.get("content", query)
        else:  # fall back to string repr
            query = getattr(last_message, "content", str(last_message))
    return query.strip()


def _invoke_prompt(prompt_name: str, **kwargs: str) -> str:
    prompt = get_prompt(prompt_name).format(**kwargs)
    llm = _get_llm()
    logger.debug("Invoking prompt '%s'", prompt_name)
    response = llm.invoke([HumanMessage(content=prompt)])
    return getattr(response, "content", "").strip()


def _tavily_search(query: str, *, max_results: int = 4) -> Dict[str, object]:
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TALVIY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Tavily API key. Set TAVILY_API_KEY in the environment.")

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
    }
    response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _format_tavily_results(result_payload: Dict[str, object], *, max_snippets: int = 3) -> str:
    results = result_payload.get("results", []) if isinstance(result_payload, dict) else []
    snippets = []
    for idx, item in enumerate(results[:max_snippets], start=1):
        title = item.get("title", "") if isinstance(item, dict) else ""
        content = item.get("content") or item.get("snippet") or ""
        url = item.get("url", "")
        snippets.append(f"[{idx}] {title}\n{content}\nSource: {url}".strip())
    return "\n\n".join(snippets)


def _build_deep_research_brief(query: str) -> str:
    return _invoke_prompt("deep_research_brief", input=query)


def _append_assistant_message(state: State, content: str, *, ensure_unique: bool = False) -> list:
    messages = list(state.get("messages", []))
    if ensure_unique and messages:
        last = messages[-1]
        last_role = getattr(last, "type", None) or getattr(last, "role", None)
        last_content = getattr(last, "content", None)
        if isinstance(last, dict):
            last_role = last.get("role")
            last_content = last.get("content")
        if last_role in {"ai", "assistant"} and last_content == content:
            return messages
    messages.append(AIMessage(content=content))
    return messages


# ---------------------------------------------------------------------------
# Custom hand-off tool
# ---------------------------------------------------------------------------


@tool
def custom_handoff_tool(
    agent_name: Annotated[str, "Name of the target agent node (e.g. 'tavily_agent')"],
    task_input: Annotated[str, "Input payload passed to the delegated agent"],
    state_update_json: Annotated[
        Optional[str],
        "Optional JSON encoded dictionary with additional state updates",
    ] = None,
) -> Command:
    """Return a LangGraph command instructing a jump to a sub-agent node."""

    update: Dict[str, object] = {"last_query": task_input}
    if state_update_json:
        try:
            extra = json.loads(state_update_json)
            if isinstance(extra, dict):
                update.update(extra)
        except json.JSONDecodeError:
            logger.warning("custom_handoff_tool received invalid JSON payload")

    return Command(goto=agent_name, update=update)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def supervisor_node(state: State) -> Command[Literal["tavily_agent", "mcp_agent", "image_agent", "__end__"]]:
    """Orchestrate between self-answering and specialised worker agents."""
    logger.debug("Supervisor invoked with state keys: %s", list(state.keys()))

    # If a worker already produced a response, surface it and end the flow.
    if not state.get("awaiting_subagent") and state.get("chat_response") and not state.get("final_response_sent"):
        messages = _append_assistant_message(state, state["chat_response"], ensure_unique=True)
        update = {
            "messages": messages,
            "final_response_sent": True,
            "last_answer": state["chat_response"],
            "awaiting_subagent": False,
        }
        return Command(goto="__end__", update=update)

    query = _extract_user_query(state)
    if not query:
        return Command(goto="__end__")

    lower_query = query.lower()
    classification: Optional[str] = None

    image_keywords = [
        "diagram",
        "image",
        "illustration",
        "infographic",
        "picture",
        "visual",
        "flowchart",
        "mind map",
    ]
    if any(keyword in lower_query for keyword in image_keywords):
        classification = "image_generation"

    deep_keywords = [
        "in-depth",
        "comprehensive",
        "strategy",
        "market analysis",
        "research plan",
        "detailed report",
        "systematic review",
        "compare",
    ]
    if classification is None and any(keyword in lower_query for keyword in deep_keywords):
        classification = "deep_research"

    quick_keywords = [
        "latest",
        "recent",
        "current",
        "today",
        "price",
        "news",
        "update",
        "find",
    ]
    if classification is None and any(keyword in lower_query for keyword in quick_keywords):
        classification = "quick_search"

    if classification is None:
        classification = _invoke_prompt("router", input=query).lower()
        logger.debug("Router LLM classification for query '%s': %s", query, classification)
    else:
        logger.debug("Heuristic classification for query '%s': %s", query, classification)
    logger.info("Supervisor classified query as '%s'", classification)

    if classification not in {"simple_answer", "quick_search", "deep_research", "image_generation"}:
        classification = "quick_search"

    if classification == "simple_answer":
        answer = _invoke_prompt("simple_answer", input=query)
        update = {
            "chat_response": answer,
            "last_answer": answer,
            "messages": _append_assistant_message(state, answer),
            "query_type": "simple_answer",
            "current_task": "simple_answer",
            "final_response_sent": True,
        }
        return Command(goto="__end__", update=update)

    if classification == "quick_search":
        update = {
            "query_type": "quick_search",
            "current_task": "quick_search",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "chat_response": None,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "tavily_agent",
                "task_input": query,
                "state_update_json": json.dumps(update),
            }
        )

    if classification == "deep_research":
        research_brief = _build_deep_research_brief(query)
        update = {
            "query_type": "deep_research",
            "current_task": "deep_research",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "mcp_query": research_brief,
            "chat_response": None,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "mcp_agent",
                "task_input": research_brief,
                "state_update_json": json.dumps(update),
            }
        )

    # image generation branch
    image_prompt = _invoke_prompt("image_generation", input=query)
    update = {
        "query_type": "image_generation",
        "current_task": "image_generation",
        "awaiting_subagent": True,
        "final_response_sent": False,
        "image_prompt": image_prompt,
        "chat_response": None,
    }
    return custom_handoff_tool.invoke(
        {
            "agent_name": "image_agent",
            "task_input": image_prompt,
            "state_update_json": json.dumps(update),
        }
    )


# ---------------------------------------------------------------------------
# Worker nodes
# ---------------------------------------------------------------------------


def tavily_agent_node(state: State) -> Command[Literal["supervisor"]]:
    query = state.get("last_query", "")
    try:
        raw_results = _tavily_search(query)
        formatted = _format_tavily_results(raw_results)
        logger.debug("Tavily returned %d snippets", len(raw_results.get("results", [])))
        if formatted.strip():
            answer = _invoke_prompt("quick_search_summary", query=query, search_results=formatted)
        else:
            logger.info("No Tavily snippets found – falling back to simple answer prompt")
            answer = _invoke_prompt("simple_answer", input=query)
            raw_results = {"results": []}
    except Exception as exc:  # pragma: no cover - network failure branch
        logger.exception("Tavily search failed")
        answer = f"Quick search failed: {exc}"
        raw_results = {}

    update = state.copy()
    update.update(
        {
            "awaiting_subagent": False,
            "chat_response": answer,
            "last_answer": answer,
            "last_search_payload": raw_results,
            "current_task": "quick_search",
        }
    )
    update["messages"] = _append_assistant_message(state, answer)
    return Command(goto="supervisor", update=update)


async def mcp_agent_node(state: State) -> Command[Literal["supervisor"]]:
    query = state.get("last_query") or state.get("mcp_query") or ""
    agent = create_mcp_agent()
    try:
        result = await agent.run(query)
        answer = result if isinstance(result, str) else str(result)
    except Exception as exc:  # pragma: no cover - network failure branch
        logger.exception("MCP agent run failed")
        answer = f"Deep research failed: {exc}"

    update = state.copy()
    update.update(
        {
            "awaiting_subagent": False,
            "chat_response": answer,
            "last_answer": answer,
            "current_task": "deep_research",
        }
    )
    update["messages"] = _append_assistant_message(state, answer)
    return Command(goto="supervisor", update=update)


def image_generation_node(state: State) -> Command[Literal["supervisor"]]:
    prompt = state.get("last_query") or state.get("image_prompt") or ""
    description = state.get("image_prompt") or _invoke_prompt("image_generation", input=prompt)
    client = create_openai_client({})

    try:
        result = client.generate(description, response_modalities=["image"], max_images=1)
        if result.images:
            image = result.images[0]
            encoded = base64.b64encode(image.data).decode("utf-8")
            image_url = f"data:{image.mime_type};base64,{encoded}"
            text = result.text or "Here is the generated illustration."
        else:
            image_url = ""
            text = "Image generation completed without returning an asset."
    except Exception as exc:  # pragma: no cover - network failure branch
        logger.exception("Image generation failed", exc_info=exc)
        image_url = ""
        error_text = str(exc)
        if "content_policy_violation" in error_text:
            text = (
                "Image generation request was blocked by the provider's safety filters. "
                "Please rephrase the prompt to avoid restricted content."
            )
        else:
            text = f"Image generation failed: {exc}"

    update = state.copy()
    update.update(
        {
            "awaiting_subagent": False,
            "chat_response": text,
            "last_answer": text,
            "image_url": image_url,
            "current_task": "image_generation",
        }
    )
    update["messages"] = _append_assistant_message(state, text)
    return Command(goto="supervisor", update=update)


__all__ = [
    "custom_handoff_tool",
    "supervisor_node",
    "tavily_agent_node",
    "mcp_agent_node",
    "image_generation_node",
]
