"""Supervisory and worker nodes for the agent service orchestration."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Annotated, Dict, Literal, Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from lastminute_api.application.agent_service.prompts import get_prompt
from lastminute_api.application.agent_service.state import AgentState as State
from lastminute_api.infrastructure.llm_providers.base import get_llm_by_type
from lastminute_api.infrastructure.mcp.mcp_agent import create_agent as create_mcp_agent
from lastminute_api.infrastructure.nano_bannana.openai import create_openai_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_query(state: State) -> str:
    """Return the latest user utterance and persist it on the state."""
    query = state.get("last_query") or ""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        elif isinstance(last_message, dict) and last_message.get("role") == "user":
            query = last_message.get("content", query)
        else:
            query = getattr(last_message, "content", str(last_message))
    return query.strip()


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
# Supervisor decision schema
# ---------------------------------------------------------------------------


class SupervisorDecision(BaseModel):
    """Structured decision returned by the routing LLM."""

    action: Literal["simple_answer", "quick_search", "deep_research", "image_generation"] = Field(
        description="Selected next step"
    )
    handoff_query: str = Field(
        description="Restated or refined query to pass to the chosen worker"
    )
    rationale: str = Field(description="Short reasoning for audit logging")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def supervisor_node(state: State) -> Command[Literal["tavily_agent", "mcp_agent", "image_agent", "__end__"]]:
    """Orchestrate between self-answering and specialised worker agents."""
    logger.debug("Supervisor invoked with state keys: %s", list(state.keys()))

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

    llm = get_llm_by_type("openai")
    structured_router = llm.with_structured_output(SupervisorDecision)

    history_snippets = []
    for message in state.get("messages", [])[-4:]:
        role = getattr(message, "type", None) or getattr(message, "role", None) or "assistant"
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        history_snippets.append(f"{role}: {content}")

    supervisor_messages = [
        SystemMessage(
            content=(
                "You are the supervisor for the LastMinute multi-agent system. "
                "Choose one action: simple_answer, quick_search, deep_research, image_generation. "
                "Return a refined query for the selected worker and a concise rationale."
            )
        ),
        HumanMessage(
            content=(
                "Conversation history (last turns):\n"
                + ("\n".join(history_snippets) if history_snippets else "<none>")
                + "\n\nUser query:\n"
                + query
            )
        ),
    ]

    decision = structured_router.invoke(supervisor_messages)
    logger.info("Supervisor routed to '%s' (%s)", decision.action, decision.rationale)

    handoff_query = decision.handoff_query.strip() or query
    supervisor_notes = decision.rationale.strip()

    if decision.action == "simple_answer":
        system_content = (
            "You are the LastMinute supervisor. Provide a direct, well-structured answer. "
            "Address the user's request succinctly and reference the rationale when helpful."
        )
        prompt = get_prompt("simple_answer").format(input=handoff_query)
        user_content = prompt
        if supervisor_notes:
            user_content += f"\n\nSupervisor rationale: {supervisor_notes}"
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        answer = getattr(response, "content", "").strip()
        update = {
            "chat_response": answer,
            "last_answer": answer,
            "messages": _append_assistant_message(state, answer),
            "query_type": "simple_answer",
            "current_task": "simple_answer",
            "final_response_sent": True,
            "awaiting_subagent": False,
            "routing_notes": supervisor_notes,
        }
        return Command(goto="__end__", update=update)

    if decision.action == "quick_search":
        update = {
            "query_type": "quick_search",
            "current_task": "quick_search",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "chat_response": None,
            "routing_notes": supervisor_notes,
            "handoff_instructions": supervisor_notes,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "tavily_agent",
                "task_input": handoff_query,
                "state_update_json": json.dumps(update),
            }
        )

    if decision.action == "deep_research":
        prompt = get_prompt("deep_research_brief").format(
            input=f"{handoff_query}\n\nSupervisor notes: {supervisor_notes or 'N/A'}"
        )
        brief_response = llm.invoke([HumanMessage(content=prompt)])
        research_brief = getattr(brief_response, "content", "").strip()
        update = {
            "query_type": "deep_research",
            "current_task": "deep_research",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "mcp_query": research_brief,
            "chat_response": None,
            "routing_notes": supervisor_notes,
            "handoff_instructions": supervisor_notes,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "mcp_agent",
                "task_input": research_brief,
                "state_update_json": json.dumps(update),
            }
        )

    # image_generation
    update = {
        "query_type": "image_generation",
        "current_task": "image_generation",
        "awaiting_subagent": True,
        "final_response_sent": False,
        "image_prompt": handoff_query,
        "chat_response": None,
        "routing_notes": supervisor_notes,
        "handoff_instructions": supervisor_notes,
    }
    return custom_handoff_tool.invoke(
        {
            "agent_name": "image_agent",
            "task_input": handoff_query,
            "state_update_json": json.dumps(update),
        }
    )


# ---------------------------------------------------------------------------
# Worker nodes
# ---------------------------------------------------------------------------


def tavily_agent_node(state: State) -> Command[Literal["supervisor"]]:
    query = state.get("last_query", "")
    supervisor_notes = state.get("handoff_instructions") or state.get("routing_notes") or ""
    try:
        raw_results = _tavily_search(query)
        formatted = _format_tavily_results(raw_results)
        logger.debug("Tavily returned %d snippets", len(raw_results.get("results", [])))
        llm = get_llm_by_type("openai")
        if formatted.strip():
            system_content = (
                "You are the LastMinute quick-search specialist. Summarise findings factually."
            )
            prompt = get_prompt("quick_search_summary").format(
                query=query,
                search_results=formatted,
            )
            if supervisor_notes:
                prompt += f"\n\nSupervisor notes: {supervisor_notes}"
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            answer = getattr(response, "content", "").strip()
        else:
            logger.info("No Tavily snippets found – falling back to simple answer prompt")
            system_content = (
                "You are the LastMinute supervisor answering directly. Provide a concise explanation."
            )
            prompt = get_prompt("simple_answer").format(input=query)
            if supervisor_notes:
                prompt += f"\n\nSupervisor notes: {supervisor_notes}"
            messages = [SystemMessage(content=system_content), HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            answer = getattr(response, "content", "").strip()
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
    if state.get("handoff_instructions"):
        query = f"{query}\n\nSupervisor notes: {state['handoff_instructions']}"
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
    description = state.get("image_prompt")
    if not description:
        llm = get_llm_by_type("openai")
        system_content = (
            "You are the LastMinute visual imagination agent. Produce a vivid, safe description for image generation."
        )
        prompt_text = get_prompt("image_generation").format(input=prompt)
        notes = state.get("handoff_instructions") or state.get("routing_notes") or ""
        if notes:
            prompt_text += f"\n\nSupervisor notes: {notes}"
        messages = [SystemMessage(content=system_content), HumanMessage(content=prompt_text)]
        response = llm.invoke(messages)
        description = getattr(response, "content", "").strip()
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
