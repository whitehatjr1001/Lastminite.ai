"""Google GenAI (Gemini) Nano Banana client.

This wraps the official google-genai client with a minimal interface
suited for text and optional image responses. It supports streaming
and non-streaming interactions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, Optional

from .base import NanoBananaImage, NanoBananaResult


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Best effort coercion of integers from configuration sources."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class GeminiNanoBanana:
    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        default_modalities: list[str] | None = None,
        default_max_images: Optional[int] = None,
    ):
        # Late import to avoid hard dependency at import time
        from google import genai as _genai  # type: ignore

        self._genai = _genai
        self._client = _genai.Client(api_key=api_key)
        self._default_model = default_model
        self._default_modalities = default_modalities or ["TEXT"]
        self._default_max_images = default_max_images if default_max_images is None or default_max_images >= 0 else 0

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
        """Generate aggregated text and binary artifacts from Gemini.

        Args:
            prompt: User supplied prompt.
            model: Optional override for the Gemini model identifier.
            response_modalities: Modalities to request. Defaults to the
                client's configured modalities.
            max_images: Optional cap on binary/image artifacts captured.
                ``None`` defers to the client's configured default.
            aggregate_text: When True (default) concatenate text chunks into a
                single string. When False, text chunks are ignored.
            **kwargs: Provider specific overrides forwarded to
                :meth:`generate_stream`.
        """
        image_cap = max_images if max_images is not None else self._default_max_images
        image_cap = image_cap if image_cap is None or image_cap >= 0 else 0

        text_parts: list[str] = []
        images: list[NanoBananaImage] = []
        for event in self.generate_stream(
            prompt,
            response_modalities=response_modalities,
            model=model,
            **kwargs,
        ):
            if event.get("type") == "text":
                if aggregate_text and "text" in event:
                    text_parts.append(event["text"])
            elif event.get("type") == "binary":
                if image_cap is None or len(images) < image_cap:
                    images.append(
                        NanoBananaImage(
                            data=event.get("data", b""),
                            mime_type=event.get("mime_type", "application/octet-stream"),
                            index=int(event.get("index", len(images))),
                        )
                    )

        return NanoBananaResult(text="".join(text_parts), images=images)

    def generate_text(self, prompt: str, *, model: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a full text response by aggregating the stream."""
        result = self.generate(
            prompt,
            model=model,
            response_modalities=["TEXT"],
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
        """Stream events from Gemini.

        Yields dictionaries of the form:
          - {"type": "text", "text": str}
          - {"type": "binary", "data": bytes, "mime_type": str, "index": int}
        """
        # Late import for types
        from google.genai import types as genai_types  # type: ignore

        model_name = model or self._default_model
        modalities = response_modalities or self._default_modalities

        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=prompt)],
            )
        ]

        cfg = genai_types.GenerateContentConfig(response_modalities=modalities)

        file_index = 0
        for chunk in self._client.models.generate_content_stream(
            model=model_name, contents=contents, config=cfg
        ):
            cand = getattr(chunk, "candidates", None)
            if not cand:
                continue
            content = cand[0].content if cand[0] else None
            if not content or not getattr(content, "parts", None):
                continue

            part = content.parts[0]
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                # Binary event
                yield {
                    "type": "binary",
                    "data": inline.data,
                    "mime_type": getattr(inline, "mime_type", "application/octet-stream"),
                    "index": file_index,
                }
                file_index += 1
            else:
                # Text event
                # Some SDK versions expose "text" at chunk level as well.
                text = getattr(chunk, "text", None)
                if text:
                    yield {"type": "text", "text": text}


def create_gemini_client(conf: Dict[str, Any]) -> GeminiNanoBanana:
    """Create a Gemini Nano Banana client from config and env.

    Recognized config/env keys:
      - api_key or env GEMINI_API_KEY (required)
      - model or env GEMINI_MODEL (default: gemini-2.5-flash-image-preview)
      - response_modalities (list[str]) (default: ["TEXT"])
      - max_images or env GEMINI_MAX_IMAGES (default: None for unlimited)
    """
    api_key = conf.get("api_key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY for Gemini client.")

    model = conf.get("model") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-image-preview"
    modalities = conf.get("response_modalities") or ["TEXT"]
    max_images_conf = conf.get("max_images")
    if max_images_conf is None:
        max_images_conf = os.getenv("GEMINI_MAX_IMAGES")
    max_images = _coerce_int(max_images_conf)

    return GeminiNanoBanana(
        api_key=api_key,
        default_model=model,
        default_modalities=modalities,
        default_max_images=max_images,
    )
