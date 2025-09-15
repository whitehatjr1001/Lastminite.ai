import os

import pytest


def test_gemini_client_missing_key_raises():
    from lastminute_api.infrastructure.nano_bannana.gemini import create_gemini_client

    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    with pytest.raises(ValueError) as exc:
        _ = create_gemini_client({})
    assert "GEMINI_API_KEY" in str(exc.value)


def test_gemini_generate_text_and_stream(monkeypatch):
    from lastminute_api.infrastructure.nano_bannana.base import get_nanobanana_client, clear_cache

    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-image-preview")

    clear_cache()
    client = get_nanobanana_client("gemini")

    # Test aggregate text
    result = client.generate("Say hello", response_modalities=["TEXT", "IMAGE"], max_images=1)
    assert result.text == "Hello World"
    assert len(result.images) == 1
    assert result.images[0].mime_type == "image/png"

    # Test stream yields both text and binary when requested
    events = list(client.generate_stream("Hello", response_modalities=["TEXT", "IMAGE"]))
    # Expect 3 events based on mocked stream: text, binary, text
    assert len(events) == 3
    assert events[0]["type"] == "text" and isinstance(events[0]["text"], str)
    assert events[1]["type"] == "binary" and isinstance(events[1]["data"], (bytes, bytearray))
    assert events[2]["type"] == "text"

    # Max images of zero removes binary artifacts while keeping text
    result_no_images = client.generate(
        "Another",
        response_modalities=["TEXT", "IMAGE"],
        max_images=0,
    )
    assert result_no_images.text == "Hello World"
    assert result_no_images.images == []


def test_gemini_max_images_from_env(monkeypatch):
    from lastminute_api.infrastructure.nano_bannana.base import get_nanobanana_client, clear_cache

    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-image-preview")
    monkeypatch.setenv("GEMINI_MAX_IMAGES", "0")

    clear_cache()
    client = get_nanobanana_client("gemini")

    result = client.generate("Say hello", response_modalities=["TEXT", "IMAGE"])
    assert result.images == []


def test_factory_cache_and_clear(monkeypatch):
    from lastminute_api.infrastructure.nano_bannana.base import get_nanobanana_client, clear_cache

    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    clear_cache()
    c1 = get_nanobanana_client("gemini")
    c2 = get_nanobanana_client("gemini")
    assert c1 is c2
    clear_cache()
    c3 = get_nanobanana_client("gemini")
    assert c1 is not c3


def test_together_provider_not_implemented(monkeypatch):
    # Now implemented: test text streaming and aggregation using mocks
    from lastminute_api.infrastructure.nano_bannana.base import get_nanobanana_client, clear_cache

    monkeypatch.setenv("TOGETHER_API_KEY", "tk-test")
    clear_cache()
    client = get_nanobanana_client("together")

    result = client.generate("Say hello")
    assert result.text == "Hello World"
    assert result.images == []

    events = list(client.generate_stream("Hello", response_modalities=["TEXT"]))
    assert len(events) == 2
    assert events[0]["type"] == "text" and events[0]["text"] == "Hello"
    assert events[1]["type"] == "text" and events[1]["text"] == " World"


def test_together_image_modality_not_supported(monkeypatch):
    from lastminute_api.infrastructure.nano_bannana.base import get_nanobanana_client

    monkeypatch.setenv("TOGETHER_API_KEY", "tk-test")
    client = get_nanobanana_client("together", use_cache=False)
    with pytest.raises(NotImplementedError):
        _ = list(client.generate_stream("image please", response_modalities=["IMAGE"]))
