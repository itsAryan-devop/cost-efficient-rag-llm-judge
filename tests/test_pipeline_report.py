"""Tests for suite-level aggregation, rubric weighting, and bias metrics."""
from eval.pipeline.report import build_suite_report, compute_bias_metrics
from eval.pipeline.logger import AuditLogger
from eval.pipeline.schemas import (
    CriterionScore,
    JudgeVerdict,
    PairwiseResult,
    TestCase,
    TestSuite,
    ValidationResult,
)


def _scores(values: dict[str, int]) -> list[CriterionScore]:
    return [CriterionScore(criterion=k, score=v, rationale="r") for k, v in values.items()]


def _verdict(a: dict[str, int], b: dict[str, int], winner: str = "A") -> JudgeVerdict:
    return JudgeVerdict(
        criteria_scores_a=_scores(a),
        criteria_scores_b=_scores(b),
        rationale="r",
        winner=winner,
    )


def _result(case_id: str, ab: JudgeVerdict, ba: JudgeVerdict, final: str, bias: bool = False) -> PairwiseResult:
    return PairwiseResult(
        case_id=case_id,
        verdict_ab=ab,
        verdict_ba=ba,
        final_winner=final,
        position_bias_detected=bias,
    )


# --------------------------------------------------------------------------- #
# score_clustering_detected: now means "low variance of per-criterion scores"
# --------------------------------------------------------------------------- #
def test_score_clustering_detected_when_all_scores_identical():
    # Every criterion gets a flat 3 -> standard deviation is 0 -> clustering.
    flat = {"correctness": 3, "completeness": 3, "faithfulness": 3, "conciseness": 3, "instruction_following": 3}
    v = _verdict(flat, flat)
    results = [_result(f"c{i}", v, v, "A") for i in range(3)]
    bm = compute_bias_metrics(results, ValidationResult(method="t"))
    assert bm.score_clustering_detected is True
    assert bm.score_variance == 0.0


def test_score_clustering_not_detected_when_scores_vary():
    a = {"correctness": 5, "completeness": 4, "faithfulness": 5, "conciseness": 3, "instruction_following": 4}
    b = {"correctness": 2, "completeness": 3, "faithfulness": 1, "conciseness": 4, "instruction_following": 2}
    v = _verdict(a, b)
    results = [_result(f"c{i}", v, v, "A") for i in range(3)]
    bm = compute_bias_metrics(results, ValidationResult(method="t"))
    assert bm.score_clustering_detected is False
    assert bm.score_variance > 0.5


def test_compute_bias_metrics_probe_flags():
    flat = {"correctness": 3}
    v = _verdict(flat, flat)
    results = [_result("c1", v, v, "A")]
    val = ValidationResult(
        method="test",
        details=[
            {"probe_type": "verbosity", "passed": True},
            {"probe_type": "sycophancy", "passed": False},
        ],
    )
    bm = compute_bias_metrics(results, val)
    assert bm.verbosity_probe_passed is True
    assert bm.sycophancy_probe_passed is False


# --------------------------------------------------------------------------- #
# position_flip_rate: independent measurement of raw-winner swap
# --------------------------------------------------------------------------- #
def test_position_flip_rate_measures_raw_winner_swap_independently():
    flat = {"correctness": 3}
    # 4 cases that drive flip-rate and bias-rate to *different* values, proving
    # they measure different things (the old code returned the same number twice).
    cases = [
        # (1) Agreement: AB=A, BA=B -> BA mapped to A -> no flip, no bias.
        _result("agree1", _verdict(flat, flat, "A"), _verdict(flat, flat, "B"), "A"),
        # (2) Agreement: AB=B, BA=A -> BA mapped to B -> no flip, no bias.
        _result("agree2", _verdict(flat, flat, "B"), _verdict(flat, flat, "A"), "B"),
        # (3) Tie disagreement: AB=A, BA=Tie -> different raw winners -> flip; not "same position twice" -> no bias.
        _result("partial", _verdict(flat, flat, "A"), _verdict(flat, flat, "Tie"), "A"),
        # (4) Position-1 bias: AB=A, BA=A -> mapped to B -> flip AND bias.
        _result("biased", _verdict(flat, flat, "A"), _verdict(flat, flat, "A"), "Conflict", bias=True),
    ]
    bm = compute_bias_metrics(cases, ValidationResult(method="t"))
    # 2 of 4 raw winners changed when the order was swapped.
    assert bm.position_flip_rate == 0.5
    # Only 1 of 4 had the judge pick the same position both times.
    assert bm.position_bias_rate == 0.25
    # They are now reported as distinct numbers (the old code returned the same value).
    assert bm.position_flip_rate != bm.position_bias_rate


# --------------------------------------------------------------------------- #
# Rubric weights are actually used: weighted aggregates + pass rate
# --------------------------------------------------------------------------- #
def test_weighted_aggregates_use_rubric_weights():
    # A is great on the heavily-weighted criteria, B is great on the light ones.
    a_scores = {"correctness": 5, "completeness": 5, "faithfulness": 5, "conciseness": 1, "instruction_following": 5}
    b_scores = {"correctness": 1, "completeness": 1, "faithfulness": 1, "conciseness": 5, "instruction_following": 1}
    v = _verdict(a_scores, b_scores, winner="A")
    # Mirror in BA so per-criterion means are symmetric across orderings.
    v_ba = _verdict(b_scores, a_scores, winner="B")
    results = [_result(f"c{i}", v, v_ba, "A") for i in range(3)]

    suite = TestSuite(name="t", cases=[TestCase(id=f"c{i}", input="i", output_a="a", output_b="b") for i in range(3)])
    bm = compute_bias_metrics(results, ValidationResult(method="t"))
    report = build_suite_report(suite, results, bm, ValidationResult(method="t"), AuditLogger(path="reports/_t.jsonl"))

    assert report.weighted_score_a > report.weighted_score_b
    # Correctness criterion should have weight 2.0 surfaced in the aggregate.
    correctness = next(c for c in report.criterion_aggregates if c.criterion == "correctness")
    assert correctness.weight == 2.0
    assert correctness.mean_score_a == 5.0
    assert correctness.mean_score_b == 1.0


def test_pass_rate_uses_weighted_threshold():
    # Build 4 cases: 3 strong for A, 1 weak. Mirror BA so a/b roles are
    # swapped (this is how the real pairwise pipeline calls the judge).
    a_strong = {"correctness": 5, "completeness": 5, "faithfulness": 5, "conciseness": 4, "instruction_following": 5}
    b_weak = {"correctness": 2, "completeness": 2, "faithfulness": 2, "conciseness": 3, "instruction_following": 2}
    strong_ab = _verdict(a_strong, b_weak, winner="A")
    strong_ba = _verdict(b_weak, a_strong, winner="B")  # roles swapped
    flat2 = {"correctness": 2, "completeness": 2, "faithfulness": 2, "conciseness": 2, "instruction_following": 2}
    weak_ab = _verdict(flat2, flat2, winner="Tie")
    weak_ba = _verdict(flat2, flat2, winner="Tie")
    results = [
        _result("s1", strong_ab, strong_ba, "A"),
        _result("s2", strong_ab, strong_ba, "A"),
        _result("s3", strong_ab, strong_ba, "A"),
        _result("w1", weak_ab, weak_ba, "Tie"),
    ]
    suite = TestSuite(name="t", cases=[TestCase(id=r.case_id, input="i", output_a="a", output_b="b") for r in results])
    bm = compute_bias_metrics(results, ValidationResult(method="t"))
    report = build_suite_report(suite, results, bm, ValidationResult(method="t"), AuditLogger(path="reports/_t.jsonl"))

    assert report.pass_threshold == 4.0
    assert report.pass_rate_a == 0.75   # 3 of 4 cases meet >=4 for A
    assert report.pass_rate_b == 0.0


def test_weighted_score_breaks_win_count_ties():
    # Win counts tie (0-0, all ties), but A clearly outscores B on rubric -> winner=A.
    a_better = {"correctness": 5, "completeness": 5, "faithfulness": 5, "conciseness": 5, "instruction_following": 5}
    b_worse = {"correctness": 2, "completeness": 2, "faithfulness": 2, "conciseness": 2, "instruction_following": 2}
    v_ab = _verdict(a_better, b_worse, winner="Tie")
    v_ba = _verdict(b_worse, a_better, winner="Tie")   # roles swapped
    results = [_result(f"c{i}", v_ab, v_ba, "Tie") for i in range(3)]
    suite = TestSuite(
        name="t",
        config_a="A-cfg",
        config_b="B-cfg",
        cases=[TestCase(id=f"c{i}", input="i", output_a="a", output_b="b") for i in range(3)],
    )
    bm = compute_bias_metrics(results, ValidationResult(method="t"))
    report = build_suite_report(suite, results, bm, ValidationResult(method="t"), AuditLogger(path="reports/_t.jsonl"))
    assert report.winner == "A-cfg"
