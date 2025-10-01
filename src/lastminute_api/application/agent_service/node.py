"""Supervisory and worker nodes for the agent service orchestration."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from lastminute_api.application.agent_service.prompts import get_prompt
from lastminute_api.application.agent_service.state import AgentState as State
from lastminute_api.domain.tools.registry import create_mindmap, display_mindmap, get_mindmap, simple_mindmap
from lastminute_api.infrastructure.llm_providers.base import get_llm_by_type
from lastminute_api.infrastructure.mcp.mcp_agent import create_agent as create_mcp_agent
from lastminute_api.infrastructure.nano_bannana.base import NanoBananaResult
from lastminute_api.infrastructure.nano_bannana.gemini import generate_gemini_image_result
from lastminute_api.infrastructure.nano_bannana.openai import generate_openai_image_result

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


def _preview(text: str, *, limit: int = 200) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _extract_reference_id(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Reference:\s*([A-Za-z0-9_]+)", text)
    return match.group(1) if match else None


def _clean_text_block(text: str) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if "Invalid Format" not in line]
    return "\n".join(lines).strip()


def _parse_outline_json(text: str) -> tuple[Optional[str], List[str]]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, []
    candidate = match.group(0).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate, count=1).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            payload = json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
            return None, []
    topic = payload.get("topic") if isinstance(payload, dict) else None
    subtopics = payload.get("subtopics") if isinstance(payload, dict) else []
    if isinstance(subtopics, str):
        subtopics = [chunk.strip() for chunk in re.split(r",|;", subtopics) if chunk.strip()]
    elif not isinstance(subtopics, list):
        subtopics = []
    cleaned_subtopics = [str(item).strip() for item in subtopics if str(item).strip()]
    return (str(topic).strip() if topic else None, cleaned_subtopics[:8])


def _derive_subtopics(topic: str, *sources: str, limit: int = 8) -> List[str]:
    candidates: List[str] = []
    for source in sources:
        if not source:
            continue
        for raw_line in source.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d\-•*.() ]+", "", line)
            head = re.split(r"[:\-–]|\\u2013", line, maxsplit=1)[0].strip()
            head = re.sub(r"[^A-Za-z0-9 /&()+]", "", head)
            head = re.sub(r"\s+", " ", head).strip()
            if 3 <= len(head) <= 40:
                head = head.title()
                candidates.append(head)
    if not candidates:
        base = topic.split("(")[0].strip() or "Study"
        candidates = [
            f"{base} Overview".strip(),
            "Core Concepts",
            "Key Techniques",
            "Applications",
            "Benefits",
            "Challenges",
            "Best Practices",
            "Future Trends",
        ]
    unique: List[str] = []
    for item in candidates:
        short = item[:36].strip()
        if short and short not in unique:
            unique.append(short)
        if len(unique) >= limit:
            break
    return unique


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


def _format_tavily_results(
    result_payload: Dict[str, object], *, max_snippets: int = 3, max_chars: int = 600
) -> str:
    results = result_payload.get("results", []) if isinstance(result_payload, dict) else []
    snippets = []
    for idx, item in enumerate(results[:max_snippets], start=1):
        title = item.get("title", "") if isinstance(item, dict) else ""
        content = item.get("content") or item.get("snippet") or ""
        if isinstance(content, str) and len(content) > max_chars:
            content = content[: max_chars - 3] + "..."
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


def _recent_user_messages(state: State, limit: int = 3) -> List[str]:
    """Return the most recent user-authored messages (oldest first)."""

    collected: List[str] = []
    for message in reversed(state.get("messages", [])):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        if role in {"human", "user"} and content:
            collected.append(str(content))
            if len(collected) >= limit:
                break
    return list(reversed(collected))

def get_conversation_history(state: State) -> str:
    """Return the full conversation history as a formatted string."""
    history = []
    for message in state.get("messages", []):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        if role and content:
            history.append(f"{role.capitalize()}: {content}")
    return "\n".join(history)


def _compose_handoff_instructions(*, action: str, query: str, rationale: str, state: State) -> str:
    """Summarise the supervisor intent for a downstream worker."""

    sections: List[str] = [f"Action: {action}", f"User request: {query}"]

    if rationale:
        sections.append(f"Supervisor rationale: {rationale}")

    recent_user = _recent_user_messages(state, limit=2)
    if recent_user:
        sections.extend(["Recent user turns:", *recent_user])

    recent_answer = state.get("last_answer")
    if recent_answer:
        sections.append(f"Previous assistant summary: {_preview(str(recent_answer))}")

    return "\n".join(sections)


def _build_grounded_messages(
    *,
    system_content: str,
    state: State,
    user_query: str,
    guidance: str,
    supervisor_notes: Optional[str] = None,
    extra_sections: Optional[Sequence[Tuple[str, str]]] = None,
    include_history: bool = True,
) -> List[object]:
    """Compose grounded system/human messages with shared conventions."""

    sections: List[str] = ["User question:", user_query.strip()]

    if extra_sections:
        for title, body in extra_sections:
            body_text = body.strip()
            if body_text:
                sections.extend(["", title, body_text])

    guidance_text = guidance.strip()
    if guidance_text:
        sections.extend(["", "Instructions:", guidance_text])

    if supervisor_notes:
        sections.extend(["", f"Supervisor rationale: {supervisor_notes.strip()}"])

    if include_history:
        history_messages = _recent_user_messages(state)
        if history_messages:
            sections.extend(["", "Recent user messages:"] + history_messages)

    human_message = "\n".join(filter(None, sections))
    return [
        SystemMessage(content=system_content.strip()),
        HumanMessage(content=human_message),
    ]


def _user_requested_sources(state: State, query: str) -> bool:
    """Detect whether the user explicitly asked for sources or links."""

    keywords = {
        "source",
        "sources",
        "citation",
        "citations",
        "cite",
        "link",
        "links",
        "reference",
        "references",
    }

    def _contains_keyword(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    if _contains_keyword(query):
        return True

    for message in _recent_user_messages(state, limit=5):
        if _contains_keyword(message):
            return True

    return False


def _extract_image_options(
    state: State,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
    """Return shared/openai/gemini option dictionaries and preferred provider."""

    raw = state.get("image_options")
    if not isinstance(raw, dict):
        return {}, {}, {}, None

    provider_pref = raw.get("provider")
    provider = str(provider_pref).lower() if provider_pref else None

    shared = {k: v for k, v in raw.items() if k not in {"provider", "openai", "gemini"}}
    openai_opts = raw.get("openai") if isinstance(raw.get("openai"), dict) else {}
    gemini_opts = raw.get("gemini") if isinstance(raw.get("gemini"), dict) else {}

    return shared, dict(openai_opts), dict(gemini_opts), provider


def _result_to_payload(result: NanoBananaResult) -> Tuple[str, str]:
    """Convert a Nano Banana result into assistant text and a data URL."""

    if not result.images:
        raise RuntimeError("No image content returned.")

    image = result.images[0]
    mime_type = getattr(image, "mime_type", "image/png") or "image/png"
    encoded = base64.b64encode(image.data).decode("utf-8")
    image_url = f"data:{mime_type};base64,{encoded}"
    text = result.text or "Here is the generated illustration."
    return text, image_url


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

    action: Literal["simple_answer", "quick_search", "deep_research", "image_generation", "mind_map"] = Field(
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
        final_answer = str(state.get("chat_response", "")).strip()
        state["chat_response"] = final_answer
        logger.debug("Supervisor final answer: %s", _preview(final_answer))
        messages = _append_assistant_message(state, final_answer, ensure_unique=True)
        update = {
            "messages": messages,
            "final_response_sent": True,
            "last_answer": final_answer,
            "awaiting_subagent": False,
        }
        return Command(goto="__end__", update=update)

    query = _extract_user_query(state)
    if not query:
        return Command(goto="__end__")

    logger.info("🧭 Supervisor input: %s", _preview(query))

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

    system_content = "\n\n".join([get_prompt("supervisor_system"), get_prompt("router")])
    history_text = "\n".join(history_snippets) if history_snippets else "<none>"
    router_payload = get_prompt("router_human").format(
        history=history_text,
        user_query=query,
        notes=state.get("routing_notes", "<none>"),
    )

    supervisor_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=router_payload),
    ]

    decision = structured_router.invoke(supervisor_messages)
    logger.info("🪢 Routed to %s | notes: %s", decision.action, decision.rationale)

    handoff_query = decision.handoff_query.strip() or query
    supervisor_notes = decision.rationale.strip()

    if decision.action == "simple_answer":
        logger.debug("🗣️ Simple answer handoff query: %s", _preview(handoff_query))

        system_content = get_prompt("supervisor_system")
        guidance = get_prompt("simple_answer").format(
            question=handoff_query,
            notes=supervisor_notes or "None",
        )
        messages = _build_grounded_messages(
            system_content=system_content,
            state=state,
            user_query=handoff_query,
            guidance=guidance,
            supervisor_notes=supervisor_notes,
        )

        response = llm.invoke(messages)
        logger.debug("Supervisor simple answer: %s", _preview(getattr(response, "content", "")))
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

    if decision.action == "mind_map":
        logger.debug("🗺️ Mind map handoff query: %s", _preview(handoff_query))
        instructions = _compose_handoff_instructions(
            action="mind_map",
            query=handoff_query,
            rationale=supervisor_notes,
            state=state,
        )
        update = {
            "query_type": "mind_map",
            "current_task": "mind_map",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "chat_response": None,
            "routing_notes": supervisor_notes,
            "handoff_instructions": instructions,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "mind_map_agent",
                "task_input": handoff_query,
                "state_update_json": json.dumps(update),
            }
        )

    if decision.action == "quick_search":
        logger.debug("🔎 Quick search handoff query: %s", _preview(handoff_query))
        instructions = _compose_handoff_instructions(
            action="quick_search",
            query=handoff_query,
            rationale=supervisor_notes,
            state=state,
        )
        update = {
            "query_type": "quick_search",
            "current_task": "quick_search",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "chat_response": None,
            "routing_notes": supervisor_notes,
            "handoff_instructions": instructions,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "tavily_agent",
                "task_input": handoff_query,
                "state_update_json": json.dumps(update),
            }
        )

    if decision.action == "deep_research":
        logger.debug("🧠 Deep research handoff query: %s", _preview(handoff_query))
        prompt = get_prompt("deep_research_brief").format(
            query=handoff_query,
            rationale=supervisor_notes or "None",
        )
        brief_response = llm.invoke([HumanMessage(content=prompt)])
        research_brief = getattr(brief_response, "content", "").strip()
        instructions = _compose_handoff_instructions(
            action="deep_research",
            query=handoff_query,
            rationale=supervisor_notes,
            state=state,
        )
        update = {
            "query_type": "deep_research",
            "current_task": "deep_research",
            "awaiting_subagent": True,
            "final_response_sent": False,
            "mcp_query": research_brief,
            "chat_response": None,
            "routing_notes": supervisor_notes,
            "handoff_instructions": instructions,
        }
        return custom_handoff_tool.invoke(
            {
                "agent_name": "mcp_agent",
                "task_input": research_brief,
                "state_update_json": json.dumps(update),
            }
        )

    # image_generation
    instructions = _compose_handoff_instructions(
        action="image_generation",
        query=handoff_query,
        rationale=supervisor_notes,
        state=state,
    )
    update = {
        "query_type": "image_generation",
        "current_task": "image_generation",
        "awaiting_subagent": True,
        "final_response_sent": False,
        "image_prompt": handoff_query,
        "chat_response": None,
        "routing_notes": supervisor_notes,
        "handoff_instructions": instructions,
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
    logger.info("🔎 Tavily agent received: %s", _preview(query))
    if supervisor_notes:
        logger.debug("🔎 Supervisor notes: %s", supervisor_notes)
    try:
        raw_results = _tavily_search(query)
        formatted = _format_tavily_results(raw_results)
        logger.debug("Tavily returned %d snippets", len(raw_results.get("results", [])))
        llm = get_llm_by_type("openai")
        if formatted.strip():
            system_content = (
                "You are the LastMinute quick-search specialist. Answer using only the snippets provided. "
                "Be succinct, cite snippet indices like [1], and do not ask the user to rephrase."
            )
            guidance = get_prompt("quick_search_summary").format(
                query=query,
                search_results=formatted,
                notes=supervisor_notes or "None",
            )
            messages = _build_grounded_messages(
                system_content=system_content,
                state=state,
                user_query=query,
                guidance=guidance,
                supervisor_notes=supervisor_notes,
                extra_sections=[("Search snippets (numbered):", formatted)],
            )
            response = llm.invoke(messages)
            answer = getattr(response, "content", "").strip()
            logger.debug("Tavily agent answer: %s", _preview(answer))
        else:
            logger.info("No Tavily snippets found – falling back to simple answer prompt")
            system_content = (
                "You are the LastMinute supervisor responding without snippets. Provide the best concise answer you can, "
                "flagging any uncertainty, and do not ask the user to restate their request."
            )
            guidance = get_prompt("simple_answer").format(
                question=query,
                notes=supervisor_notes or "None",
            )
            messages = _build_grounded_messages(
                system_content=system_content,
                state=state,
                user_query=query,
                guidance=guidance,
                supervisor_notes=supervisor_notes,
            )
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
    user_query = state.get("last_query", "")
    research_brief = state.get("mcp_query", "")
    supervisor_notes = state.get("handoff_instructions") or state.get("routing_notes") or ""

    instruction_lines = [
        "Provide a structured, well-organised answer grounded in the evidence you gather.",
        "Cite any tools or documents you rely on when summarising key findings.",
    ]
    if _user_requested_sources(state, user_query or research_brief):
        instruction_lines.append(
            "Include a 'Sources' section with URLs or references if the user requested sources or links."
        )
    else:
        instruction_lines.append("If any detail is uncertain, explain the limitation and suggest follow-up steps.")

    query_sections: List[str] = []
    if user_query.strip():
        query_sections.extend(["User request:", user_query.strip()])
    if research_brief.strip() and research_brief.strip() != user_query.strip():
        query_sections.extend(["", "Research brief:", research_brief.strip()])
    if supervisor_notes:
        query_sections.extend(["", f"Supervisor rationale: {supervisor_notes.strip()}"])

    history = _recent_user_messages(state, limit=5)
    if history:
        query_sections.extend(["", "Recent user messages:"] + history)

    instruction_text = "\n".join(f"- {line}" for line in instruction_lines)
    query_sections.extend(["", "Instructions:", instruction_text])

    composed_query = "\n".join(filter(None, query_sections)) or research_brief or user_query
    logger.info("🧠 MCP agent received: %s", _preview(composed_query))

    agent = create_mcp_agent()
    try:
        result = await agent.run(composed_query)
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
            "last_research_payload": answer,
        }
    )
    update["messages"] = _append_assistant_message(state, answer)
    return Command(goto="supervisor", update=update)


def mind_map_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """Generate a grounded mind map and return a lightweight reference."""

    topic = (state.get("mind_map_instruction") or state.get("last_query") or "").strip()
    supervisor_notes = (state.get("handoff_instructions") or state.get("routing_notes") or "").strip()

    logger.info("🗺️ Mind map agent received: %s", _preview(topic or supervisor_notes))

    search_payload: Dict[str, object] = {}
    search_notes = ""
    search_query = topic or supervisor_notes

    if search_query:
        try:
            search_payload = _tavily_search(search_query, max_results=6)
            search_notes = _format_tavily_results(search_payload, max_snippets=5, max_chars=1000)
        except Exception as exc:  # pragma: no cover - network failure branch
            logger.warning("Mind map Tavily search failed: %s", exc)

    knowledge_block = search_notes or "No external snippets were available; rely on user context and prior turns."

    llm = get_llm_by_type("openai")

    outline_prompt = (
        "Summarise the topic and notes into JSON with keys 'topic' and 'subtopics'. "
        "Return only JSON. Limit subtopics to concise phrases (<=4 words)."
    )
    outline_messages = [
        SystemMessage(content=outline_prompt),
        HumanMessage(
            content=(
                f"Topic: {topic or 'Not provided'}\n"
                f"Supervisor notes: {supervisor_notes or 'None'}\n"
                "Supporting knowledge:\n"
                f"{knowledge_block}"
            )
        ),
    ]

    outline_topic: Optional[str] = None
    outline_subtopics: List[str] = []
    try:
        outline_response = llm.invoke(outline_messages)
        outline_text = _clean_text_block(getattr(outline_response, "content", ""))
        outline_topic, outline_subtopics = _parse_outline_json(outline_text)
    except Exception:  # pragma: no cover - outline failure
        logger.debug("Mind map outline generation failed", exc_info=True)

    topic_name = outline_topic or topic or search_query or "Study Topic"
    subtopics = outline_subtopics or _derive_subtopics(
        topic_name,
        supervisor_notes,
        knowledge_block,
    )

    serialized = ", ".join(subtopics[:8]) or topic_name

    try:
        result_text = str(create_mindmap.invoke(f"{topic_name}: {serialized}")).strip()
    except Exception as exc:  # pragma: no cover - fallback to simple mind map
        logger.warning("create_mindmap failed, retrying with simple_mindmap", exc_info=True)
        try:
            result_text = str(
                simple_mindmap.invoke(
                    {
                        "topic": topic_name,
                        "subtopics": ",".join(subtopics[:6]),
                    }
                )
            ).strip()
        except Exception as final_exc:  # pragma: no cover - both tools failed
            logger.exception("Mind map generation failed")
            result_text = f"Mind map generation failed: {final_exc}"

    result_text = _clean_text_block(result_text)
    mind_map_reference = _extract_reference_id(result_text)
    mind_map_summary = result_text
    mind_map_url: Optional[str] = None

    if mind_map_reference:
        try:
            display_mindmap(mind_map_reference)
        except Exception:  # pragma: no cover - filesystem guard
            logger.debug("Mind map display helper failed for %s", mind_map_reference)
        mind_map_url = get_mindmap(mind_map_reference)
        if isinstance(mind_map_url, str) and mind_map_url.startswith("Reference not found"):
            mind_map_url = None

        overview_lines = [
            f"Here's the study outline for {topic_name}:",
            "",
        ]
        for item in subtopics[:8]:
            overview_lines.append(f"- {item}")

        overview_lines.append("")
        overview_lines.append(f"Reference ID for later viewing: {mind_map_reference}")
        mind_map_summary = "\n".join(overview_lines).strip()

    if not mind_map_summary:
        mind_map_summary = "Unable to generate a mind map summary at this time."

    update = {
        "awaiting_subagent": False,
        "chat_response": mind_map_summary,
        "last_answer": mind_map_summary,
        "current_task": "mind_map",
        "query_type": "mind_map",
        "mind_map_instruction": topic,
        "mind_map_summary": mind_map_summary,
        "mind_map_reference": mind_map_reference,
        "mind_map_url": mind_map_url,
        "image_url": mind_map_url,
        "last_search_payload": search_payload,
        "routing_notes": supervisor_notes,
        "handoff_instructions": None,
        "final_response_sent": False,
    }
    update["messages"] = _append_assistant_message(state, mind_map_summary)
    return Command(goto="supervisor", update=update)



def image_generation_node(state: State) -> Command[Literal["supervisor"]]:
    user_query = state.get("last_query", "")
    base_prompt = state.get("image_prompt") or user_query
    supervisor_notes = state.get("handoff_instructions") or state.get("routing_notes") or ""

    logger.info("🎨 Image agent received: %s", _preview(base_prompt))

    llm = get_llm_by_type("openai")
    system_content = (
        "You are the LastMinute visual imagination agent. Craft a single vivid, safe prompt for an educational diagram. "
        "Do not ask the user for more input. Mention key entities, relationships, colours, and layout in one paragraph."
    )
    guidance = get_prompt("image_generation").format(
        prompt=base_prompt,
        notes=supervisor_notes or "None",
    )
    messages = _build_grounded_messages(
        system_content=system_content,
        state=state,
        user_query=base_prompt,
        guidance=guidance,
        supervisor_notes=supervisor_notes,
        include_history=True,
    )

    response = llm.invoke(messages)
    description = getattr(response, "content", "").strip() or base_prompt

    shared_opts, openai_opts, gemini_opts, provider_pref = _extract_image_options(state)

    image_url = ""
    text = ""
    used_provider: Optional[str] = None
    last_error: Optional[str] = None

    def _try_gemini() -> Tuple[str, str]:
        result = generate_gemini_image_result(
            description,
            shared_options=shared_opts,
            provider_options=gemini_opts,
        )
        return _result_to_payload(result)

    def _try_openai() -> Tuple[str, str]:
        result = generate_openai_image_result(
            description,
            shared_options=shared_opts,
            provider_options=openai_opts,
        )
        return _result_to_payload(result)

    if provider_pref == "openai":
        try:
            text, image_url = _try_openai()
            used_provider = "openai"
        except Exception as exc:  # pragma: no cover - provider-specific failure
            last_error = str(exc)
            logger.error("OpenAI image generation failed: %s", exc, exc_info=exc)
    else:
        try:
            text, image_url = _try_gemini()
            used_provider = "gemini"
        except Exception as exc:  # pragma: no cover - provider-specific failure
            last_error = str(exc)
            logger.warning("Gemini image generation failed: %s", exc, exc_info=exc)
            try:
                text, image_url = _try_openai()
                used_provider = "openai"
            except Exception as fallback_exc:  # pragma: no cover - provider-specific failure
                last_error = str(fallback_exc)
                logger.error("OpenAI fallback failed: %s", fallback_exc, exc_info=fallback_exc)

    if not image_url:
        text = text or f"Image generation failed: {last_error or 'No provider available.'}"

    update = state.copy()
    update.update(
        {
            "awaiting_subagent": False,
            "chat_response": text,
            "last_answer": text,
            "image_url": image_url,
            "current_task": "image_generation",
            "image_prompt": description,
            "image_provider": used_provider,
        }
    )
    update["messages"] = _append_assistant_message(state, text)
    return Command(goto="supervisor", update=update)


__all__ = [
    "custom_handoff_tool",
    "supervisor_node",
    "tavily_agent_node",
    "mcp_agent_node",
    "mind_map_agent_node",
    "image_generation_node",
]
