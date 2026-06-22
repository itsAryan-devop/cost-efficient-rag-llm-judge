from src.generation import NO_CONTEXT_MESSAGE, generate_answer


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
