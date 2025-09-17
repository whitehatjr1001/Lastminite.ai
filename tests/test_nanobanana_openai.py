# tests/test_nanobanana_openai.py
# Live API tests for the OpenAI Nano Banana provider.
# Requires: pip install openai pytest python-dotenv
# Env: see .env.example in repo root.

from __future__ import annotations
import os
import time
import base64
import pytest

# Import your provider factory directly (preferred for black-box testing)
from lastminute_api.infrastructure.nano_bannana.openai import create_openai_client  # type: ignore

# Small helper to build conf from env
def _conf():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "text_model": os.getenv("OPENAI_TEXT_MODEL", "gpt-5-mini"),
        "image_model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        "size": os.getenv("OPENAI_IMAGE_SIZE", "1024x1024"),
        "quality": os.getenv("OPENAI_IMAGE_QUALITY", "high"),
        "background": os.getenv("OPENAI_IMAGE_BACKGROUND", "transparent"),
        "max_n": int(os.getenv("OPENAI_IMAGE_MAX_N", "2")),
    }

@pytest.mark.timeout(45)
def test_generate_text_gpt5_under_30s():
    client = create_openai_client(_conf())
    t0 = time.time()
    text = client.generate_text("One-line summary of CRISPR mechanism for exam revision.")
    elapsed = time.time() - t0
    assert isinstance(text, str) and len(text.strip()) > 0
    assert elapsed < float(os.getenv("OPENAI_TARGET_SECONDS", "30"))

@pytest.mark.timeout(60)
def test_generate_image_gpt_image_1_png_bytes():
    client = create_openai_client(_conf())
    res = client.generate(
        "Simple labeled diagram of glycolysis suitable for undergraduate revision, clean labels, high contrast.",
        response_modalities=["image"],
        n=1,  # can override per-call
    )
    assert res.images, "No image returned"
    img = res.images
    assert img.mime_type == "image/png"
    assert isinstance(img.data, (bytes, bytearray)) and len(img.data) > 10_000  # ~sanity threshold

@pytest.mark.timeout(60)
def test_text_and_image_combo_single_call():
    client = create_openai_client(_conf())
    res = client.generate(
        "Generate 3 bullet points on mitochondrial respiration and a simple schematic diagram.",
        response_modalities=["text", "image"],
        n=1,
    )
    assert isinstance(res.text, str) and len(res.text.strip()) > 0
    assert res.images and res.images.mime_type == "image/png"

@pytest.mark.timeout(45)
def test_streaming_text_gpt5_accumulates_tokens():
    client = create_openai_client(_conf())
    chunks = []
    for event in client.generate_stream("List 3 steps of PCR in short bullets."):
        etype = event.get("type", "")
        data = event.get("data", {})
        if etype.endswith(".delta"):
            # Responses API streams deltas; exact keying may vary by SDK version
            # Extract any available text-like piece from payload
            txt = data.get("delta", {}).get("content", "") or data.get("delta", {}).get("text", "")
            if isinstance(txt, str):
                chunks.append(txt)
        elif etype.endswith("completed"):
            break
    full = "".join(chunks).strip()
    assert len(full) > 0

# Optional: quick vision sanity (image understanding) using text path.
# Your Nano Banana interface currently sends a plain string prompt to Responses.
# This probes vision semantically by referencing a public image URL in the prompt.
@pytest.mark.timeout(45)
def test_semantic_vision_prompt_text_mode():
    client = create_openai_client(_conf())
    prompt = "Describe the main anatomical features visible in this image: https://upload.wikimedia.org/wikipedia/commons/3/3a/Gray1201.png"
    text = client.generate_text(prompt)
    assert isinstance(text, str) and len(text.strip()) > 0
