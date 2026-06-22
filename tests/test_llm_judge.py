import json

import pytest

from eval.llm_judge import (
    _parse_answer_judge_response,
    _parse_graded_score,
    evaluate_answer,
)
from src.config import settings


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
