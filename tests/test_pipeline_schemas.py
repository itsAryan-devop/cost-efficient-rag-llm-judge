import pytest
from eval.pipeline.schemas import (
    TestCase, TestSuite, JudgeVerdict, CriterionScore, PairwiseResult, BiasMetrics,
)


def test_test_case_minimal():
    tc = TestCase(id="t1", input="hello", output_a="a", output_b="b")
    assert tc.id == "t1"
    assert tc.gold_winner is None


def test_test_suite_from_dict():
    data = {
        "name": "test",
        "cases": [
            {"id": "1", "input": "q", "output_a": "a", "output_b": "b"},
        ],
    }
    suite = TestSuite.model_validate(data)
    assert len(suite.cases) == 1


def test_judge_verdict_validates():
    v = JudgeVerdict(
        criteria_scores_a=[CriterionScore(criterion="c", score=5, rationale="good")],
        criteria_scores_b=[CriterionScore(criterion="c", score=3, rationale="ok")],
        rationale="A is better",
        winner="A",
    )
    assert v.winner == "A"


def test_judge_verdict_rejects_invalid_winner():
    with pytest.raises(Exception):
        JudgeVerdict(
            criteria_scores_a=[],
            criteria_scores_b=[],
            rationale="",
            winner="X",
        )


def test_bias_metrics_defaults():
    bm = BiasMetrics(position_bias_rate=0.1, position_flip_rate=0.2)
    assert bm.self_enhancement_warning is False
