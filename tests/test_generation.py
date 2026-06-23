from src.generation import NO_CONTEXT_MESSAGE, generate_answer
from src.config import settings
from src.generation import GenerationResult


class RateLimitError(Exception):
    status_code = 429


def test_generate_answer_skips_llm_when_no_context():
    result = generate_answer("What is this?", [])

    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.skipped_llm is True
    assert result.token_usage == 0
    assert result.provider
    assert result.model


def test_generate_answer_skips_llm_when_distance_is_too_high():
    result = generate_answer(
        "Who won the football match?",
        [{"id": "chunk-1", "text": "Unrelated context", "_distance": 99.0}],
    )

    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.skipped_llm is True


def test_generate_answer_falls_back_from_gemini_to_groq(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "gemini")
    monkeypatch.setattr(settings, "generation_fallback_provider", "groq")
    monkeypatch.setattr("src.generation._gemini_generate", lambda _prompt: (_ for _ in ()).throw(RateLimitError()))
    monkeypatch.setattr(
        "src.generation._groq_generate",
        lambda _prompt: GenerationResult(answer="fallback answer [chunk-1]", provider="groq", model="llama-test"),
    )

    result = generate_answer(
        "question unique fallback",
        [{"id": "chunk-1", "text": "Relevant context", "_distance": 0.1}],
        use_cache=False,
    )

    assert result.provider == "groq"
    assert "fallback answer" in result.answer
