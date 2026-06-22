import json

import pytest

from eval.llm_judge import (
    _parse_answer_judge_response,
    _parse_judge_response,
    evaluate_answer_detail,
    evaluate_faithfulness_detail,
)
from src.config import settings


def test_parse_judge_response_reads_final_score_and_rationale():
    score, rationale = _parse_judge_response(
        "The answer is supported by the provided context.\nSCORE: 1"
    )

    assert score == 1
    assert "supported" in rationale


def test_parse_judge_response_rejects_missing_score():
    with pytest.raises(ValueError):
        _parse_judge_response("Looks good to me.")


def test_parse_answer_judge_response_reads_json_scores_and_rationales():
    raw = json.dumps(
        {
            "faithfulness_score": 1,
            "faithfulness_rationale": "All claims are supported by the cited chunks.",
            "relevance_score": 0,
            "relevance_rationale": "The answer addresses a different question.",
        }
    )

    faith_score, faith_reason, relevance_score, relevance_reason = _parse_answer_judge_response(raw)

    assert faith_score == 1
    assert "supported" in faith_reason
    assert relevance_score == 0
    assert "different question" in relevance_reason


def test_parse_answer_judge_response_accepts_fenced_json_with_trailing_comma():
    raw = """```json
{
  "faithfulness_score": 1,
  "faithfulness_rationale": "The answer is grounded.",
  "relevance_score": 1,
  "relevance_rationale": "The answer addresses the query.",
}
```"""

    faith_score, faith_reason, relevance_score, relevance_reason = _parse_answer_judge_response(raw)

    assert faith_score == 1
    assert faith_reason == "The answer is grounded."
    assert relevance_score == 1
    assert relevance_reason == "The answer addresses the query."


def test_parse_answer_judge_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        _parse_answer_judge_response("faithfulness_score: 1")


def test_mock_judge_returns_rationale(monkeypatch):
    monkeypatch.setattr(settings, "judge_provider", "mock")

    result = evaluate_faithfulness_detail("context", "answer")

    assert result.score == 1
    assert result.rationale
    assert result.raw_response.endswith("SCORE: 1")


def test_mock_answer_judge_returns_both_scores(monkeypatch):
    monkeypatch.setattr(settings, "judge_provider", "mock")

    result = evaluate_answer_detail("query", "context", "answer")

    assert result.faithfulness_score == 1
    assert result.faithfulness_rationale
    assert result.relevance_score == 1
    assert result.relevance_rationale
