"""LLM provider base and factory utilities.

This module provides a simple, environment-driven factory for creating
chat LLM instances using LangChain integrations (OpenAI and Groq).

Configuration can be supplied via environment variables using a flexible
prefix pattern:

  - For role-based configs (e.g., "basic", "tools", "prompt"):
      {ROLE}_MODEL__{KEY}=value
      Example: BASIC_MODEL__provider=openai
               BASIC_MODEL__model=gpt-4o-mini

  - Or direct provider types:
      OPENAI_MODEL__model=gpt-4o-mini
      GROQ_MODEL__model=llama-3.1-8b-instant

If no provider is specified, the role name (llm_type) is used as provider
when it is one of: "openai", "groq".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Simple in-process cache by logical LLM type (e.g., "basic", "openai").
_llm_cache: Dict[str, Any] = {}


def _get_env_model_conf(llm_type: str) -> Dict[str, Any]:
    """Collect configuration from environment variables for a given type.

    Uses the prefix "{TYPE}_MODEL__". Keys are lower-cased after the prefix.
    Example envs:
      BASIC_MODEL__provider=openai
      BASIC_MODEL__model=gpt-4o-mini
      OPENAI_MODEL__api_key=sk-...
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    if conf:
        logger.info(
            "Loaded %s conf from env: %s",
            llm_type,
            {k: "***" if "key" in k else v for k, v in conf.items()},
        )
    return conf


def get_llm_by_type(llm_type: str, *, use_cache: bool = True) -> Any:
    """Return a chat LLM instance for the given logical type.

    llm_type can be a role (e.g., "basic", "tools", "prompt") or a provider
    name ("openai", "groq"). Configuration is read from env via
    ``{TYPE}_MODEL__`` variables. A ``provider`` key may be provided to map a
    role to a concrete backend provider.
    """
    if use_cache and llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = _get_env_model_conf(llm_type)
    provider = (conf.get("provider") or llm_type).lower()

    if provider == "openai":
        from .chat_openai import create_openai_chat

        model = create_openai_chat(conf)
    elif provider == "groq":
        from .chat_groq import create_groq_chat

        model = create_groq_chat(conf)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Expected one of: 'openai', 'groq'."
        )

    if use_cache:
        _llm_cache[llm_type] = model
    return model


def clear_llm_cache() -> None:
    """Clear the in-process LLM cache."""
    _llm_cache.clear()
