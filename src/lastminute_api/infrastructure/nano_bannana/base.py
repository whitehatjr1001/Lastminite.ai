"""Nano Banana inference base and factory.

This package provides a thin abstraction over multimodal inference
providers used for quick STEM revision features (e.g., generating text
and images from prompts). It focuses on a clean interface and late
imports for optional dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NanoBananaImage(BaseModel):
    """Container for binary artifacts returned by a Nano Banana provider."""

    data: bytes
    mime_type: str
    index: int


class NanoBananaResult(BaseModel):
    """Structured response from a Nano Banana generation request."""

    text: str = ""
    images: List[NanoBananaImage] = Field(default_factory=list)


class NanoBananaClient(Protocol):
    """Protocol for Nano Banana multimodal clients."""

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_modalities: list[str] | None = None,
        max_images: Optional[int] = None,
        aggregate_text: bool = True,
        **kwargs: Any,
    ) -> NanoBananaResult:  # pragma: no cover - protocol
        ...

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:  # pragma: no cover - protocol
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        response_modalities: list[str] | None = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[dict, None, None]:  # pragma: no cover - protocol
        ...


_cache: Dict[str, Any] = {}


def get_nanobanana_client(provider: str = "gemini", conf: Optional[Dict[str, Any]] = None, *, use_cache: bool = True) -> NanoBananaClient:
    """Return a Nano Banana client for the given provider.

    - provider: currently supports "gemini" (Google GenAI). Hooks for "together" are provided.
    - conf: optional dict with provider-specific configuration.
    - use_cache: cache instances per provider key.
    """
    key = provider.lower()
    if use_cache and key in _cache:
        return _cache[key]

    if key == "gemini":
        from .gemini import create_gemini_client

        client = create_gemini_client(conf or {})
    elif key == "together":
        from .together import create_together_client

        client = create_together_client(conf or {})
    else:
        raise ValueError(f"Unsupported Nano Banana provider: {provider}")

    if use_cache:
        _cache[key] = client
    return client


def clear_cache() -> None:
    _cache.clear()
