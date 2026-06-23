import json
import pytest
from eval.pipeline.config import pipeline_settings
from eval.pipeline.judge import _dispatch_call, _extract_json, _parse_verdict
from eval.pipeline.schemas import JudgeVerdict


def _make_valid_json():
    return json.dumps({
        "criteria_scores_a": [{"criterion": "correctness", "score": 5, "rationale": "perfect"}],
        "criteria_scores_b": [{"criterion": "correctness", "score": 3, "rationale": "ok"}],
        "rationale": "A is better",
        "winner": "A",
    })


def test_extract_json_plain():
    raw = _make_valid_json()
    assert json.loads(_extract_json(raw))


def test_extract_json_with_markdown_fences():
    raw = f"```json\n{_make_valid_json()}\n```"
    assert json.loads(_extract_json(raw))


def test_extract_json_with_surrounding_text():
    raw = f"Here is my evaluation:\n{_make_valid_json()}\nDone."
    assert json.loads(_extract_json(raw))


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError):
        _extract_json("No JSON here at all.")


def test_extract_json_trailing_comma():
    raw = '{"winner": "A", "rationale": "good",}'
    result = _extract_json(raw)
    assert '"winner"' in result


def test_parse_verdict_valid():
    raw = _make_valid_json()
    verdict = _parse_verdict(raw)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.winner == "A"


def test_parse_verdict_invalid_json_raises():
    with pytest.raises((ValueError, json.JSONDecodeError, Exception)):
        _parse_verdict("this is not json")


class RateLimitError(Exception):
    status_code = 429


def test_dispatch_call_falls_back_from_gemini_to_groq(monkeypatch):
    monkeypatch.setattr(pipeline_settings, "judge_provider", "gemini")
    monkeypatch.setattr(pipeline_settings, "judge_model", "gemini-test")
    monkeypatch.setattr(pipeline_settings, "judge_fallback_provider", "groq")
    monkeypatch.setattr(pipeline_settings, "judge_fallback_model", "groq-test")
    monkeypatch.setattr("eval.pipeline.judge._call_gemini", lambda *_args: (_ for _ in ()).throw(RateLimitError()))
    monkeypatch.setattr("eval.pipeline.judge._call_groq", lambda *_args: (_make_valid_json(), 10, 20))

    raw, prompt_tokens, completion_tokens, provider, model = _dispatch_call("prompt", "a", "b")

    assert json.loads(raw)["winner"] == "A"
    assert prompt_tokens == 10
    assert completion_tokens == 20
    assert provider == "groq"
    assert model == "groq-test"
