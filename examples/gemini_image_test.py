"""Standalone Gemini image-generation smoke test.

This mirrors ``examples/generate_image_test.py`` but exercises the
Nano Banana Gemini client instead of OpenAI. It is handy for quickly
verifying that credentials, prompt scaffolding, and binary decoding are
working before integrating the Gemini path into the wider agent graph.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

from lastminute_api.infrastructure.nano_bannana.gemini import create_gemini_client


PROMPT = dedent(
    """
    Render a clean, vector-style mind map illustrating the major steps of the
    CRISPR-Cas9 gene editing workflow. Use a dark teal background, bright node
    labels, and thin connector lines. Highlight key nodes for DNA targeting,
    guide RNA, Cas9 cleavage, and repair pathways.
    """
).strip()

OUTPUT_FILENAME = Path("gemini_test_image.png")


def generate_and_save_gemini_image() -> None:
    """Generate an image with Gemini and persist it locally."""

    print("--- Running Gemini Image Generation Test ---")

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("🔴 ERROR: GEMINI_API_KEY not found. Check your .env file.")
        return

    print("✅ Environment key loaded. Initializing Gemini client...")

    try:
        client = create_gemini_client(
            {
                "response_modalities": ["TEXT", "IMAGE"],
                "max_images": os.getenv("GEMINI_MAX_IMAGES"),
            }
        )

        model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-image-preview"
        max_images_env = os.getenv("GEMINI_MAX_IMAGES")
        try:
            max_images = max(1, int(max_images_env)) if max_images_env else 1
        except ValueError:
            max_images = 1

        print(f"✅ Client ready. Sending prompt to Gemini:\n   '{PROMPT}'")
        print("   (This may take up to ~30 seconds depending on the model.)")

        result = client.generate(
            PROMPT,
            model=model,
            response_modalities=["TEXT", "IMAGE"],
            max_images=max_images,
        )

        if not result.images:
            print("🔴 ERROR: Gemini did not return an image.")
            if result.text:
                print("ℹ️ Gemini text response:\n", result.text)
            return

        image_data = result.images[0].data
        OUTPUT_FILENAME.write_bytes(image_data)

        print(f"\n🎉 SUCCESS! Image saved as '{OUTPUT_FILENAME.name}'.")
        if result.text:
            print("ℹ️ Gemini text response snippet:\n", result.text[:400])

    except Exception as exc:  # pragma: no cover - manual smoke test
        print(f"\n🔴 An unexpected error occurred: {exc}")
        print("   Please verify your API key, billing status, and network connectivity.")


if __name__ == "__main__":
    generate_and_save_gemini_image()

