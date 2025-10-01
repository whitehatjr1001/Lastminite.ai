"""Chainlit entrypoint for the LastMinute supervisory agent service."""

from __future__ import annotations

import base64
import logging
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

from lastminute_api.application.agent_service.service import (
    configure_agent_logging,
    run_revision_agent,
    summarise_agent_result,
)


LOGGER = logging.getLogger(__name__)


def _decode_data_uri(uri: str) -> tuple[str | None, bytes | None]:
    if not uri.startswith("data:"):
        return None, None
    try:
        header, payload = uri.split(",", 1)
        mime = header.split(";")[0].removeprefix("data:")
        blob = base64.b64decode(payload)
        return mime or "image/png", blob
    except Exception:  # pragma: no cover - defensive catch
        LOGGER.exception("Failed to decode data URI from image agent output")
        return None, None


def _coerce_history(messages: List[BaseMessage] | None) -> List[BaseMessage]:
    return list(messages) if messages else []


@cl.on_chat_start
async def on_chat_start() -> None:
    load_dotenv()
    configure_agent_logging(logging.DEBUG)
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "👋 Hi! I’m the LastMinute supervisor. Ask me complex questions, request quick facts, "
            "or even describe an image you’d like generated."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    history: List[BaseMessage] = _coerce_history(cl.user_session.get("history"))
    try:
        state = await run_revision_agent(message.content, history=history)
    except Exception as exc:  # pragma: no cover - surfacing errors to UI
        LOGGER.exception("Supervisor run failed")
        await cl.Message(content=f"⚠️ Something went wrong: {exc}").send()
        return

    summary = summarise_agent_result(state)
    answer = summary.get("answer") or "I couldn’t produce an answer this time."
    await cl.Message(content=answer).send()

    display_url = summary.get("image_url") or summary.get("mind_map_url")
    if display_url:
        mime, blob = _decode_data_uri(display_url)
        if blob:
            image_element = cl.Image(name="Generated image", content=blob, mime=mime or "image/png")
        else:
            image_element = cl.Image(name="Generated image", url=display_url)
        await cl.Message(content="Here is the generated image:", elements=[image_element]).send()

    if state.get("routing_notes"):
        await cl.Message(
            author="Supervisor",
            content=f"_Routing notes:_ {state['routing_notes']}",
            disable_feedback=True,
        ).send()

    updated_history = state.get("messages") or []
    cl.user_session.set("history", updated_history)
