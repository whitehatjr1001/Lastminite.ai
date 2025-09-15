import os

import pytest


def test_openai_factory_with_role_and_cache(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    # Configure a role that maps to OpenAI
    monkeypatch.setenv("BASIC_MODEL__provider", "openai")
    monkeypatch.setenv("BASIC_MODEL__model", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    base.clear_llm_cache()
    llm1 = base.get_llm_by_type("basic")
    llm2 = base.get_llm_by_type("basic")

    # Instance is from mocked ChatOpenAI and cached
    assert getattr(llm1, "kwargs", None) is not None
    assert llm1 is llm2
    assert llm1.kwargs["model"] == "gpt-4o-mini"


def test_openai_missing_api_key_raises(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    monkeypatch.setenv("BASIC_MODEL__provider", "openai")
    monkeypatch.setenv("BASIC_MODEL__model", "gpt-4o-mini")
    # Ensure no API key is present
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    base.clear_llm_cache()
    with pytest.raises(ValueError) as exc:
        _ = base.get_llm_by_type("basic", use_cache=False)
    assert "OpenAI API key" in str(exc.value)


def test_openai_kwargs_coercion(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    monkeypatch.setenv("OPENAI_MODEL__model", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_MODEL__temperature", "0.7")
    monkeypatch.setenv("OPENAI_MODEL__max_tokens", "256")
    monkeypatch.setenv("OPENAI_MODEL__timeout", "5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example")
    monkeypatch.setenv("OPENAI_ORG", "org_123")

    base.clear_llm_cache()
    llm = base.get_llm_by_type("openai", use_cache=False)

    assert llm.kwargs["model"] == "gpt-4o-mini"
    assert isinstance(llm.kwargs["temperature"], float) and llm.kwargs["temperature"] == pytest.approx(0.7)
    assert isinstance(llm.kwargs["max_tokens"], int) and llm.kwargs["max_tokens"] == 256
    assert isinstance(llm.kwargs["timeout"], float) and llm.kwargs["timeout"] == pytest.approx(5.0)
    assert llm.kwargs["base_url"] == "https://api.openai.example"
    assert llm.kwargs["organization"] == "org_123"


def test_groq_factory_direct_provider(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    monkeypatch.setenv("GROQ_MODEL__model", "llama-3.1-8b-instant")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")

    base.clear_llm_cache()
    llm = base.get_llm_by_type("groq", use_cache=False)

    assert getattr(llm, "kwargs", None) is not None
    assert llm.kwargs["model"] == "llama-3.1-8b-instant"


def test_groq_missing_api_key_raises(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    monkeypatch.setenv("GROQ_MODEL__model", "llama-3.1-8b-instant")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    base.clear_llm_cache()
    with pytest.raises(ValueError) as exc:
        _ = base.get_llm_by_type("groq", use_cache=False)
    assert "Groq API key" in str(exc.value)


def test_clear_llm_cache(monkeypatch):
    from lastminute_api.infrastructure.llm_providers import base

    monkeypatch.setenv("OPENAI_MODEL__model", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    llm1 = base.get_llm_by_type("openai")
    base.clear_llm_cache()
    llm2 = base.get_llm_by_type("openai")

    assert llm1 is not llm2

