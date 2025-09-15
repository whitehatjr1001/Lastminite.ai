import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


# Ensure the `src` directory is on sys.path for imports like `lastminute_api.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, x):
        return {"content": "ok"}


class _FakeGroq:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, x):
        return {"content": "ok"}


@pytest.fixture(autouse=True)
def mock_llm_modules(monkeypatch):
    """Mock third-party LLM modules so tests don't require external packages."""
    # Mock langchain_openai
    openai_mod = ModuleType("langchain_openai")
    openai_mod.ChatOpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

    # Mock langchain_groq
    groq_mod = ModuleType("langchain_groq")
    groq_mod.ChatGroq = _FakeGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", groq_mod)

    yield


@pytest.fixture(autouse=True)
def mock_google_genai(monkeypatch):
    """Mock google.genai and google.genai.types for Gemini client tests."""
    # Build types submodule
    types_mod = ModuleType("google.genai.types")

    class _TContent:
        def __init__(self, role: str, parts: list):
            self.role = role
            self.parts = parts

    class _TPart:
        def __init__(self, text: str = "", inline_data=None):
            self.text = text
            self.inline_data = inline_data

        @classmethod
        def from_text(cls, text: str):
            return cls(text=text)

    class _TGenCfg:
        def __init__(self, response_modalities=None):
            self.response_modalities = response_modalities or ["TEXT"]

    types_mod.Content = _TContent
    types_mod.Part = _TPart
    types_mod.GenerateContentConfig = _TGenCfg

    # Build genai module
    genai_mod = ModuleType("google.genai")

    class _InlineData:
        def __init__(self, data: bytes, mime_type: str = "application/octet-stream"):
            self.data = data
            self.mime_type = mime_type

    class _Chunk:
        def __init__(self, *, text: str | None = None, binary: bytes | None = None, mime_type: str = "image/png"):
            # Build the candidate/content/part structure expected by code
            class _Cand:
                def __init__(self, part):
                    class _Cont:
                        def __init__(self, parts):
                            self.parts = parts

                    self.content = _Cont([part])

            if binary is not None:
                part = _TPart(inline_data=_InlineData(binary, mime_type))
            else:
                part = _TPart()

            self.candidates = [_Cand(part)]
            self.text = text

    class _Models:
        def generate_content_stream(self, model=None, contents=None, config=None):  # noqa: D401 - mock
            # Yields two text chunks and one binary chunk
            yield _Chunk(text="Hello")
            yield _Chunk(binary=b"\x89PNG\r\n", mime_type="image/png")
            yield _Chunk(text=" World")

    class _Client:
        def __init__(self, api_key: str):  # noqa: D401 - mock
            self.api_key = api_key
            self.models = _Models()

    genai_mod.Client = _Client

    # Build google package and attach genai + types
    google_pkg = ModuleType("google")
    # Register modules in sys.modules
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    # Also set attribute for from google import genai
    setattr(google_pkg, "genai", genai_mod)

    yield


@pytest.fixture(autouse=True)
def mock_together(monkeypatch):
    """Mock the Together SDK and its streaming interface."""
    together_mod = ModuleType("together")

    class _Delta:
        def __init__(self, content: str | None):
            self.content = content

    class _Choice:
        def __init__(self, content: str | None):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, text: str | None = None):
            # Provide both possible access paths
            self.choices = [_Choice(text)]
            self.text = text

    class _ChatCompletions:
        def create(self, *, model: str, messages: list, stream: bool = False):  # noqa: D401 - mock
            assert stream is True

            def _gen():
                yield _Chunk("Hello")
                yield _Chunk(" World")

            return _gen()

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class _Together:
        def __init__(self, api_key: str):  # noqa: D401 - mock
            self.api_key = api_key
            self.chat = _Chat()

    together_mod.Together = _Together
    monkeypatch.setitem(sys.modules, "together", together_mod)

    yield


@pytest.fixture(autouse=True)
def clean_env_and_cache(monkeypatch):
    """Clean relevant environment variables and provider cache between tests."""
    keys_to_clear = [
        # role-based prefixes
        "BASIC_MODEL__provider",
        "BASIC_MODEL__model",
        "BASIC_MODEL__temperature",
        "BASIC_MODEL__timeout",
        "BASIC_MODEL__max_tokens",
        "TOOLS_MODEL__provider",
        "TOOLS_MODEL__model",
        # direct provider prefixes
        "OPENAI_MODEL__model",
        "OPENAI_MODEL__temperature",
        "OPENAI_MODEL__timeout",
        "OPENAI_MODEL__max_tokens",
        "GROQ_MODEL__model",
        "GROQ_MODEL__temperature",
        "GROQ_MODEL__timeout",
        "GROQ_MODEL__max_tokens",
        # API keys and base URLs
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_TEMPERATURE",
        "OPENAI_ORG",
        "GROQ_API_KEY",
        "GROQ_BASE_URL",
        "GROQ_TEMPERATURE",
        # gemini
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        # together
        "TOGETHER_API_KEY",
        "TOGETHER_MODEL",
    ]
    original_env = {k: os.environ.get(k) for k in keys_to_clear}
    try:
        for k in keys_to_clear:
            if k in os.environ:
                monkeypatch.delenv(k, raising=False)

        # Clear provider cache
        from lastminute_api.infrastructure.llm_providers.base import clear_llm_cache

        clear_llm_cache()
        yield
    finally:
        for k, v in original_env.items():
            if v is not None:
                monkeypatch.setenv(k, v)
