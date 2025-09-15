"""Together AI Nano Banana client.

This wraps the official Together SDK with a minimal interface aligned
with the Nano Banana protocol used in this project.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, Optional

from .base import NanoBananaResult


class TogetherNanoBanana:
    def __init__(self, *, api_key: str, default_model: str):
        # Late import to avoid hard dependency at import time
        from together import Together  # type: ignore

        self._Together = Together
        self._client = Together(api_key=api_key)
        self._default_model = default_model

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_modalities: list[str] | None = None,
        max_images: Optional[int] = None,
        aggregate_text: bool = True,
        **kwargs: Any,
    ) -> NanoBananaResult:
        """Generate a Together response, aggregating supported modalities."""
        modalities = response_modalities or ["TEXT"]
        if any(m for m in modalities if m.upper() != "TEXT"):
            raise NotImplementedError("Together client currently supports TEXT modality only.")

        text_parts: list[str] = []
        if aggregate_text:
            for event in self.generate_stream(
                prompt,
                response_modalities=modalities,
                model=model,
                **kwargs,
            ):
                if event.get("type") == "text" and "text" in event:
                    text_parts.append(event["text"])
        else:
            # Exhaust the stream to honour prompt execution even when
            # aggregation is disabled.
            for _ in self.generate_stream(
                prompt,
                response_modalities=modalities,
                model=model,
                **kwargs,
            ):
                continue

        return NanoBananaResult(text="".join(text_parts), images=[])

    def generate_text(self, prompt: str, *, model: Optional[str] = None, **kwargs: Any) -> str:
        result = self.generate(
            prompt,
            model=model,
            response_modalities=["TEXT"],
            aggregate_text=True,
            max_images=0,
            **kwargs,
        )
        return result.text

    def generate_stream(
        self,
        prompt: str,
        *,
        response_modalities: list[str] | None = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[dict, None, None]:
        """Stream text events from Together chat completions.

        Current implementation yields text deltas only. Image modalities
        are not supported via this path and will raise NotImplementedError
        if requested.
        """
        modalities = response_modalities or ["TEXT"]
        if any(m for m in modalities if m.upper() != "TEXT"):
            raise NotImplementedError("Together client currently supports TEXT modality only.")

        model_name = model or self._default_model

        # Use streaming chat completions
        # Compatible with Together SDK patterns; tests mock this behavior.
        stream = self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        # Stream yields objects with delta content or text property depending on SDK version
        for chunk in stream:
            # Try "choices[0].delta.content" (OpenAI-compatible streaming)
            try:
                choices = getattr(chunk, "choices", None)
                if choices and getattr(choices[0], "delta", None):
                    delta = choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield {"type": "text", "text": content}
                    continue
            except Exception:
                pass

            # Fallback to a generic text attribute
            text = getattr(chunk, "text", None)
            if text:
                yield {"type": "text", "text": text}


def create_together_client(conf: Dict[str, Any]) -> TogetherNanoBanana:
    """Create a Together Nano Banana client from config and env.

    Recognized config/env keys:
      - api_key or env TOGETHER_API_KEY (required)
      - model or env TOGETHER_MODEL (default: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
    """
    api_key = conf.get("api_key") or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing TOGETHER_API_KEY for Together client.")

    model = conf.get("model") or os.getenv("TOGETHER_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    return TogetherNanoBanana(api_key=api_key, default_model=model)
