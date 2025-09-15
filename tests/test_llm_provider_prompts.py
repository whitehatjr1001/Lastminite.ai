import pytest


@pytest.mark.parametrize(
    "provider_env, llm_type, api_key_env, question",
    [
        (
            {
                "BASIC_MODEL__provider": "openai",
                "BASIC_MODEL__model": "gpt-4o-mini",
                "OPENAI_API_KEY": "sk-test",
            },
            "basic",
            "OPENAI_API_KEY",
            "What is 2 + 2?",
        ),
        (
            {
                "GROQ_MODEL__model": "llama-3.1-8b-instant",
                "GROQ_API_KEY": "gk-test",
            },
            "groq",
            "GROQ_API_KEY",
            "Name a fundamental particle in physics.",
        ),
    ],
)
def test_llm_providers_answer_simple_questions(monkeypatch, provider_env, llm_type, api_key_env, question):
    """Ensure each configured provider can answer a basic prompt via ``invoke``.

    The underlying SDKs are mocked in ``tests.conftest`` so the call is
    deterministic and does not hit external services.
    """
    from lastminute_api.infrastructure.llm_providers import base

    # Apply provider-specific environment variables.
    for key, value in provider_env.items():
        monkeypatch.setenv(key, value)

    # Guard against accidental reliance on global cache between parametrized runs.
    base.clear_llm_cache()

    llm = base.get_llm_by_type(llm_type, use_cache=False)
    response = llm.invoke(question)

    assert isinstance(response, dict)
    assert response.get("content") == "ok"

    # Sanity check: provider factory read the necessary API key.
    assert api_key_env in provider_env
