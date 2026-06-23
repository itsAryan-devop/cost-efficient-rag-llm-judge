import json

import pytest

from eval.llm_judge import (
    _parse_answer_judge_response,
    _parse_graded_score,
    evaluate_answer,
)
from src.config import settings


class RateLimitError(Exception):
    status_code = 429


def test_parse_answer_judge_response_reads_graded_scores_and_rationales():
    raw = json.dumps(
        {
            "faithfulness_score": 5,
            "faithfulness_rationale": "All claims are supported by the cited chunks.",
            "relevance_score": 2,
            "relevance_rationale": "The answer mostly addresses a different question.",
        }
    )
    faith, faith_reason, rel, rel_reason = _parse_answer_judge_response(raw)
    assert faith == 5
    assert "supported" in faith_reason
    assert rel == 2
    assert "different question" in rel_reason


def test_parse_answer_judge_response_accepts_fenced_json_with_trailing_comma():
    raw = """```json
{
  "faithfulness_score": 4,
  "faithfulness_rationale": "Grounded.",
  "relevance_score": 5,
  "relevance_rationale": "On topic.",
}
```"""
    faith, _, rel, _ = _parse_answer_judge_response(raw)
    assert faith == 4
    assert rel == 5


def test_parse_answer_judge_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        _parse_answer_judge_response("faithfulness_score: 5")


def test_parse_graded_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        _parse_graded_score(0, "faithfulness_score")
    with pytest.raises(ValueError):
        _parse_graded_score(6, "faithfulness_score")
    assert _parse_graded_score(3, "faithfulness_score") == 3


def test_mock_judge_scores_grounded_answer_high(monkeypatch):
    monkeypatch.setattr(settings, "judge_provider", "mock")
    context = (
        "502 Bad Gateway means the server, acting as a gateway or proxy, received an invalid "
        "response from the upstream server."
    )
    query = "What does HTTP status code 502 Bad Gateway mean?"
    answer = "502 Bad Gateway means a gateway or proxy received an invalid response from the upstream server."
    result = evaluate_answer(query, context, answer)
    assert result.faithfulness_score >= 4
    assert result.provider == "mock"


def test_mock_judge_scores_planted_wrong_answer_low(monkeypatch):
    """The whole point of the judge: a confidently-wrong answer must score low."""
    monkeypatch.setattr(settings, "judge_provider", "mock")
    context = (
        "502 Bad Gateway means the server, acting as a gateway or proxy, received an invalid "
        "response from the upstream server."
    )
    query = "What does HTTP status code 502 Bad Gateway mean?"
    wrong = "502 Bad Gateway means the client supplied invalid login credentials and must authenticate again."
    result = evaluate_answer(query, context, wrong)
    assert result.faithfulness_score <= 2


def test_mock_judge_scores_verbose_unsupported_answer_low(monkeypatch):
    monkeypatch.setattr(settings, "judge_provider", "mock")
    context = "502 Bad Gateway means a proxy received an invalid response from the upstream server."
    query = "What does HTTP status code 502 Bad Gateway mean?"
    verbose = (
        "The 502 status is a profound meditation on the ephemeral nature of distributed consensus, "
        "entropy, and the cosmic ballet of asynchronous existence."
    )
    result = evaluate_answer(query, context, verbose)
    assert result.faithfulness_score <= 2


def test_real_judge_falls_back_from_gemini_to_groq(monkeypatch):
    monkeypatch.setattr(settings, "judge_provider", "gemini")
    monkeypatch.setattr(settings, "judge_fallback_provider", "groq")
    monkeypatch.setattr(settings, "judge_model", "")
    monkeypatch.setattr(settings, "generation_model", "gemini-test")
    monkeypatch.setattr(settings, "groq_model", "groq-test")

    def fake_call(provider, model, prompt):
        if provider == "gemini":
            raise RateLimitError()
        return json.dumps(
            {
                "faithfulness_score": 5,
                "faithfulness_rationale": "Fallback judge found the answer grounded.",
                "relevance_score": 5,
                "relevance_rationale": "Fallback judge found the answer relevant.",
            }
        )

    monkeypatch.setattr("eval.llm_judge._call_judge_model", fake_call)

    result = evaluate_answer("unique fallback query", "context supports answer", "answer")

    assert result.provider == "groq"
    assert result.model == "groq-test"
    assert result.faithfulness_score == 5
