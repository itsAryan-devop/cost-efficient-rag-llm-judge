import pytest
from eval.pipeline.schemas import JudgeVerdict, CriterionScore
from eval.pipeline.pairwise import _resolve_winner


def _verdict(winner: str) -> JudgeVerdict:
    return JudgeVerdict(
        criteria_scores_a=[CriterionScore(criterion="c", score=4, rationale="ok")],
        criteria_scores_b=[CriterionScore(criterion="c", score=3, rationale="ok")],
        rationale="test",
        winner=winner,
    )


def test_both_agree_a_wins():
    # AB: A wins, BA: B wins (BA's B = Asst2 = original A)
    winner, bias = _resolve_winner(_verdict("A"), _verdict("B"))
    assert winner == "A"
    assert bias is False


def test_both_agree_b_wins():
    # AB: B wins, BA: A wins (BA's A = Asst1 = original B)
    winner, bias = _resolve_winner(_verdict("B"), _verdict("A"))
    assert winner == "B"
    assert bias is False


def test_position_bias_always_first():
    # Both pick Assistant 1
    winner, bias = _resolve_winner(_verdict("A"), _verdict("A"))
    assert winner == "Conflict"
    assert bias is True


def test_position_bias_always_second():
    # Both pick Assistant 2
    winner, bias = _resolve_winner(_verdict("B"), _verdict("B"))
    assert winner == "Conflict"
    assert bias is True


def test_both_tie():
    winner, bias = _resolve_winner(_verdict("Tie"), _verdict("Tie"))
    assert winner == "Tie"
    assert bias is False


def test_one_tie_one_decisive():
    winner, bias = _resolve_winner(_verdict("A"), _verdict("Tie"))
    # ba Tie maps to Tie, ab says A -> lean A
    assert winner == "A"
    assert bias is False
