# src/lastminute_api/infrastructure/nano_bannana/openai.py

from __future__ import annotations
import base64, os
from typing import Any, Dict, Generator, List, Optional
from .base import NanoBananaImage, NanoBananaResult

from openai import OpenAI  # pip install openai>=1

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}  # portrait/landscape/square [17][19]
ALLOWED_QUALITY = {"low", "medium", "high", "standard", "hd"}  # fidelity vs speed tradeoff [1][12]
ALLOWED_BG = {"transparent", "opaque"}  # background toggle [1][8]
ALLOWED_STYLES = {"vivid", "natural"}

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")  # override via conf/kwargs [21]
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")  # current image API [4][21]
DEFAULT_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")  # safe default [17][19]
DEFAULT_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "high")  # fidelity [1][12]
DEFAULT_IMAGE_BACKGROUND = os.getenv("OPENAI_IMAGE_BACKGROUND", "transparent")  # PNG alpha [1][8]

class _OpenAIClientImpl:
    def __init__(self, conf: Dict[str, Any]) -> None:
        api_key = conf.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Defaults (can be overridden per-call)
        self.text_model = conf.get("text_model", DEFAULT_TEXT_MODEL)  # [21]
        self.image_model = conf.get("image_model", DEFAULT_IMAGE_MODEL)  # [21]
        self.size = conf.get("size", DEFAULT_IMAGE_SIZE)  # [17]
        self.quality = conf.get("quality", DEFAULT_IMAGE_QUALITY)  # [1]
        self.background = conf.get("background", DEFAULT_IMAGE_BACKGROUND)  # [1]
        self.max_n = conf.get("max_n", 4)  # gpt-image-1 typically 1–4 [8]

    def _resolve_image_opts(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        size = kwargs.pop("size", self.size)
        if size not in ALLOWED_SIZES:
            size = self.size  # guard invalid [17][19]

        quality = kwargs.pop("quality", self.quality)
        if quality not in ALLOWED_QUALITY:
            quality = self.quality  # [1][12]

        background = kwargs.pop("background", self.background)
        if background not in ALLOWED_BG:
            background = self.background  # [1][8]

        n = int(kwargs.pop("n", 1))
        n = max(1, min(n, int(self.max_n)))  # 1–4 typical [8]

        # force base64 so we can return bytes reliably
        response_format = kwargs.pop("response_format", "b64_json")  # [8][19]
        style = kwargs.pop("style", None)
        if style and style not in ALLOWED_STYLES:
            style = None

        resolved: Dict[str, Any] = {
            "size": size,
            "quality": quality,
            "background": background,
            "n": n,
            "response_format": response_format,
        }
        if style:
            resolved["style"] = style
        return resolved
    # DELETE your old 'generate' function and REPLACE it with this one.

    def _build_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """OpenAI Responses API expects content describing the user input."""
        return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_modalities: List[str] | None = None,
        max_images: Optional[int] = None,
        aggregate_text: bool = True,
        **kwargs: Any,
    ) -> NanoBananaResult:
        """
        Uses the new unified `responses.create` endpoint for both text and images.
        This is the correct method for models like gpt-image-1.
        """
        modes = {m.lower() for m in (response_modalities or ["text"])}
        out = NanoBananaResult()

        wants_image = "image" in modes
        wants_text = ("text" in modes) or (not wants_image)

        call_kwargs: Dict[str, Any] = dict(kwargs)
        aggregate = call_kwargs.pop("aggregate_text", aggregate_text)

        image_opts: Dict[str, Any] = {}
        if wants_image:
            image_opts = self._resolve_image_opts(call_kwargs)
            if max_images is not None:
                try:
                    requested = int(max_images)
                except (TypeError, ValueError):
                    requested = 1
                image_opts["n"] = max(1, min(requested, int(image_opts.get("n", 1))))

        if wants_text:
            text_request: Dict[str, Any] = {
                "model": (model or self.text_model),
                "input": self._build_prompt(prompt),
            }
            text_request.update(call_kwargs)
            text_resp = self.client.responses.create(**text_request)

            text_parts: List[str] = []
            for item in getattr(text_resp, "output", []) or []:
                if getattr(item, "type", "") == "output_text" and hasattr(item, "text"):
                    text_parts.append(item.text)

            if text_parts:
                out.text = "".join(text_parts) if aggregate else text_parts[0]

        if wants_image:
            image_request: Dict[str, Any] = {
                "model": (model or self.image_model),
                "prompt": prompt,
            }
            image_request.update(image_opts)

            image_model_name = str(image_request.get("model", "")).lower()
            if "dall-e-3" in image_model_name:
                image_request.pop("background", None)
                image_request["n"] = 1
                quality = image_request.get("quality")
                if quality not in {"standard", "hd"}:
                    image_request["quality"] = "standard"

            image_resp = self.client.images.generate(**image_request)
            data_items = getattr(image_resp, "data", []) or []
            for idx, item in enumerate(data_items):
                b64_payload = getattr(item, "b64_json", None)
                if b64_payload:
                    blob = base64.b64decode(b64_payload)
                    out.images.append(
                        NanoBananaImage(
                            data=blob,
                            mime_type=getattr(item, "mime_type", "image/png"),
                            index=len(out.images),
                        )
                    )

        return out

    def generate_text(self, prompt: str, *, model: Optional[str] = None, **kwargs: Any) -> str:
        resp = self.client.responses.create(
            model=(model or self.text_model),
            input=self._build_prompt(prompt),
            **kwargs,
        )
        text_parts = [getattr(item, "text", "") for item in getattr(resp, "output", []) if getattr(item, "type", "") == "output_text"]
        return "".join(text_parts)

    def generate_stream(
        self,
        prompt: str,
        *,
        response_modalities: List[str] | None = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[dict, None, None]:
        stream_kwargs: Dict[str, Any] = {
            "model": (model or self.text_model),
            "input": self._build_prompt(prompt),
        }
        stream_kwargs.update(kwargs)

        with self.client.responses.stream(**stream_kwargs) as stream:  # [10]
            for event in stream:
                yield {"type": getattr(event, "type", "unknown"), "data": event.model_dump(exclude_none=True)}  # [10]

def create_openai_client(conf: Dict[str, Any]) -> _OpenAIClientImpl:
    return _OpenAIClientImpl(conf or {})  # factory hook [16]
