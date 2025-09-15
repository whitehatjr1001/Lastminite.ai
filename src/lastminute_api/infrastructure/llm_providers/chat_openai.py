"""OpenAI chat provider using LangChain's ChatOpenAI."""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _coerce_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def create_openai_chat(conf: Dict[str, Any]) -> ChatOpenAI:
    """Create a ChatOpenAI instance from provided configuration.

    Supported keys in ``conf`` (strings are accepted and coerced):
      - provider: ignored here (used by factory)
      - api_key: defaults to env ``OPENAI_API_KEY``
      - model: defaults to env ``OPENAI_MODEL`` or ``gpt-4o-mini``
      - base_url: optional override for API base URL
      - temperature: float, defaults to 0
      - max_tokens: int, optional
      - timeout: seconds, optional (coerced to float)
      - organization: optional
    """
    api_key = conf.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or provide 'api_key'.")

    model = conf.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    temperature = _coerce_float(conf.get("temperature", os.getenv("OPENAI_TEMPERATURE", 0)), 0.0)
    max_tokens = _coerce_int(conf.get("max_tokens"))
    timeout = _coerce_float(conf.get("timeout"))
    base_url = conf.get("base_url") or os.getenv("OPENAI_BASE_URL")
    organization = conf.get("organization") or os.getenv("OPENAI_ORG")

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature if temperature is not None else 0.0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization

    return ChatOpenAI(**kwargs)
