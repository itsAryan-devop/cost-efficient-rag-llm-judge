"""Suite-level aggregation and report output."""
from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime, timezone

from .config import pipeline_settings
from .logger import AuditLogger
from .prompts import DEFAULT_RUBRIC
from .schemas import (
    BiasMetrics,
    CriterionAggregate,
    PairwiseResult,
    SuiteReport,
    TestSuite,
    ValidationResult,
)


# --------------------------------------------------------------------------- #
# Per-case weighted score (rubric weights are USED, not decorative)
# --------------------------------------------------------------------------- #
def _weighted_score(scores: list, weights: dict[str, float]) -> float:
    """Return sum(score * weight) / sum(weight) for the criteria present.

    `scores` is a list of CriterionScore-like objects with `.criterion` and `.score`.
    Falls back to a simple mean if none of the criteria match the weights map.
    """
    total_w = 0.0
    total = 0.0
    for s in scores:
        w = weights.get(s.criterion, 0.0)
        if w <= 0:
            continue
        total_w += w
        total += s.score * w
    if total_w == 0:
        nums = [s.score for s in scores]
        return sum(nums) / len(nums) if nums else 0.0
    return total / total_w


def _case_weighted_scores(result: PairwiseResult, weights: dict[str, float]) -> tuple[float, float]:
    """Return (weighted_A, weighted_B) averaged across the AB and BA orderings.

    In the AB ordering, `criteria_scores_a` are A's scores; in the BA ordering, the
    roles are swapped, so `criteria_scores_a` are B's. Averaging both orderings is
    what makes the aggregate position-symmetric.
    """
    a_ab = _weighted_score(result.verdict_ab.criteria_scores_a, weights)
    b_ab = _weighted_score(result.verdict_ab.criteria_scores_b, weights)
    # In BA: criteria_scores_a belongs to B, criteria_scores_b belongs to A.
    b_ba = _weighted_score(result.verdict_ba.criteria_scores_a, weights)
    a_ba = _weighted_score(result.verdict_ba.criteria_scores_b, weights)
    return (a_ab + a_ba) / 2.0, (b_ab + b_ba) / 2.0


def _per_criterion_means(
    results: list[PairwiseResult],
    rubric=DEFAULT_RUBRIC,
) -> list[CriterionAggregate]:
    """Compute mean score per criterion, per assistant, averaging both orderings."""
    aggregates: list[CriterionAggregate] = []
    for criterion in rubric.criteria:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for r in results:
            # AB ordering: a-scores are A, b-scores are B.
            for s in r.verdict_ab.criteria_scores_a:
                if s.criterion == criterion.name:
                    a_vals.append(float(s.score))
            for s in r.verdict_ab.criteria_scores_b:
                if s.criterion == criterion.name:
                    b_vals.append(float(s.score))
            # BA ordering: a-scores are B, b-scores are A.
            for s in r.verdict_ba.criteria_scores_a:
                if s.criterion == criterion.name:
                    b_vals.append(float(s.score))
            for s in r.verdict_ba.criteria_scores_b:
                if s.criterion == criterion.name:
                    a_vals.append(float(s.score))
        cases_scored = min(len(a_vals), len(b_vals)) // 2  # two scores per case per assistant
        aggregates.append(
            CriterionAggregate(
                criterion=criterion.name,
                weight=criterion.weight,
                mean_score_a=round(sum(a_vals) / len(a_vals), 3) if a_vals else 0.0,
                mean_score_b=round(sum(b_vals) / len(b_vals), 3) if b_vals else 0.0,
                cases_scored=cases_scored,
            )
        )
    return aggregates


def _all_criterion_scores(results: list[PairwiseResult]) -> list[float]:
    """Flat list of every per-criterion score recorded across the suite (both orderings, both assistants)."""
    out: list[float] = []
    for r in results:
        for v in (r.verdict_ab, r.verdict_ba):
            out.extend(s.score for s in v.criteria_scores_a)
            out.extend(s.score for s in v.criteria_scores_b)
    return out


def compute_bias_metrics(
    results: list[PairwiseResult],
    validation: ValidationResult,
) -> BiasMetrics:
    """Compute aggregate bias metrics from pairwise results and validation probes."""
    total = len(results) if results else 1
    position_bias_count = sum(1 for r in results if r.position_bias_detected)

    # position_flip_rate: independent measurement of how often the RAW winner
    # changes when the order is swapped, regardless of how we resolved the case.
    # Map BA back to original A/B labels before comparing.
    ba_map = {"A": "B", "B": "A", "Tie": "Tie", "Error": "Error"}
    flips = 0
    countable = 0
    for r in results:
        w_ab = r.verdict_ab.winner
        w_ba_mapped = ba_map.get(r.verdict_ba.winner, r.verdict_ba.winner)
        if w_ab in ("Error",) or w_ba_mapped in ("Error",):
            continue
        countable += 1
        if w_ab != w_ba_mapped:
            flips += 1
    flip_rate = flips / countable if countable else 0.0

    # Validation probe outcomes
    verbosity_passed = None
    sycophancy_passed = None
    for detail in validation.details:
        if detail.get("probe_type") == "verbosity":
            verbosity_passed = detail.get("passed", False)
        if detail.get("probe_type") == "sycophancy":
            sycophancy_passed = detail.get("passed", False)

    # Self-enhancement check (judge and generator are the same provider family)
    self_enhancement = (
        pipeline_settings.judge_provider.lower() == pipeline_settings.generator_provider.lower()
    )

    # True score clustering: variance of per-criterion scores across the suite.
    all_scores = _all_criterion_scores(results)
    if len(all_scores) >= 2:
        variance = statistics.pvariance(all_scores)
        stdev = statistics.pstdev(all_scores)
    else:
        variance, stdev = 0.0, 0.0
    clustering = stdev < 0.5 and len(all_scores) >= 4  # judge is bunching scores in a narrow band

    return BiasMetrics(
        position_bias_rate=round(position_bias_count / total, 4),
        position_flip_rate=round(flip_rate, 4),
        verbosity_probe_passed=verbosity_passed,
        sycophancy_probe_passed=sycophancy_passed,
        self_enhancement_warning=self_enhancement,
        score_variance=round(variance, 4),
        score_clustering_detected=clustering,
    )


def build_suite_report(
    suite: TestSuite,
    results: list[PairwiseResult],
    bias_metrics: BiasMetrics,
    validation: ValidationResult,
    audit_logger: AuditLogger,
    pass_threshold: float = 4.0,
) -> SuiteReport:
    """Aggregate per-case results into a suite-level report."""
    total = len(results) if results else 1
    a_wins = sum(1 for r in results if r.final_winner == "A")
    b_wins = sum(1 for r in results if r.final_winner == "B")
    ties = sum(1 for r in results if r.final_winner == "Tie")
    conflicts = sum(1 for r in results if r.final_winner == "Conflict")
    errors = sum(1 for r in results if r.final_winner == "Error")

    total_valid = total - errors if total > errors else 1
    win_rate_a = round(a_wins / total_valid, 4)
    win_rate_b = round(b_wins / total_valid, 4)

    # ---- Rubric-weighted aggregates -----------------------------------------
    weights = {c.name: c.weight for c in DEFAULT_RUBRIC.criteria}
    per_criterion = _per_criterion_means(results, DEFAULT_RUBRIC)
    total_weight = sum(c.weight for c in DEFAULT_RUBRIC.criteria) or 1.0
    weighted_a = sum(c.mean_score_a * c.weight for c in per_criterion) / total_weight
    weighted_b = sum(c.mean_score_b * c.weight for c in per_criterion) / total_weight

    per_case_a, per_case_b = [], []
    for r in results:
        if r.final_winner == "Error":
            continue
        wa, wb = _case_weighted_scores(r, weights)
        per_case_a.append(wa)
        per_case_b.append(wb)
    pass_a = sum(1 for v in per_case_a if v >= pass_threshold) / len(per_case_a) if per_case_a else 0.0
    pass_b = sum(1 for v in per_case_b if v >= pass_threshold) / len(per_case_b) if per_case_b else 0.0

    # ---- Winner declaration ------------------------------------------------
    # Prefer the win-count winner. If win counts tie but weighted scores differ
    # meaningfully (>0.1), break the tie by weighted score so the rubric weights
    # actually influence the suite-level decision.
    if a_wins > b_wins:
        winner = suite.config_a
    elif b_wins > a_wins:
        winner = suite.config_b
    elif abs(weighted_a - weighted_b) >= 0.1:
        winner = suite.config_a if weighted_a > weighted_b else suite.config_b
    else:
        winner = "Tie"

    pt, ct, _ = audit_logger.total_tokens()

    return SuiteReport(
        suite_name=suite.name,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        config_a=suite.config_a,
        config_b=suite.config_b,
        total_cases=len(results),
        win_rate_a=win_rate_a,
        win_rate_b=win_rate_b,
        tie_rate=round(ties / total, 4),
        conflict_rate=round(conflicts / total, 4),
        error_rate=round(errors / total, 4),
        winner=winner,
        criterion_aggregates=per_criterion,
        weighted_score_a=round(weighted_a, 3),
        weighted_score_b=round(weighted_b, 3),
        pass_threshold=pass_threshold,
        pass_rate_a=round(pass_a, 4),
        pass_rate_b=round(pass_b, 4),
        bias_metrics=bias_metrics,
        validation=validation,
        judge_provider=pipeline_settings.judge_provider,
        judge_model=pipeline_settings.judge_model,
        generator_provider=pipeline_settings.generator_provider,
        generator_model=pipeline_settings.generator_model,
        total_judge_calls=audit_logger.total_calls(),
        total_prompt_tokens=pt,
        total_completion_tokens=ct,
        total_cost=audit_logger.total_cost(),
        results=results,
    )


def write_report(report: SuiteReport) -> None:
    """Write the suite report as JSON and CSV."""
    path = pipeline_settings.report_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"Report saved: {path}")

    csv_path = pipeline_settings.csv_report_path
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "verdict_ab_winner", "verdict_ba_winner", "final_winner",
            "position_bias", "gold_winner",
        ])
        for r in report.results:
            writer.writerow([
                r.case_id, r.verdict_ab.winner, r.verdict_ba.winner, r.final_winner,
                r.position_bias_detected, r.gold_winner or "",
            ])
    print(f"CSV saved: {csv_path}")


def print_summary(report: SuiteReport) -> None:
    """Print a human-readable summary to console."""
    print("\n" + "=" * 60)
    print(f"  LLM-as-Judge Evaluation Report: {report.suite_name}")
    print("=" * 60)
    print(f"  Config A: {report.config_a}")
    print(f"  Config B: {report.config_b}")
    print(f"  Judge: {report.judge_provider}/{report.judge_model}")
    print(f"  Generator: {report.generator_provider}/{report.generator_model}")
    print("-" * 60)
    print(f"  Total cases:        {report.total_cases}")
    print(f"  Win rate A / B:     {report.win_rate_a:.1%} / {report.win_rate_b:.1%}")
    print(f"  Tie / Conflict / Error: {report.tie_rate:.1%} / {report.conflict_rate:.1%} / {report.error_rate:.1%}")
    print(f"  Weighted score A/B: {report.weighted_score_a:.2f} / {report.weighted_score_b:.2f}  (rubric-weighted 1-5)")
    print(f"  Pass rate (>={report.pass_threshold}) A/B: {report.pass_rate_a:.1%} / {report.pass_rate_b:.1%}")
    print(f"  >>> WINNER: {report.winner} <<<")
    print("-" * 60)
    print("  Per-criterion means (A vs B):")
    for c in report.criterion_aggregates:
        print(f"    {c.criterion:<24} weight {c.weight:>3.1f}   {c.mean_score_a:.2f}  vs  {c.mean_score_b:.2f}")
    print("-" * 60)
    bm = report.bias_metrics
    print(f"  Position bias rate: {bm.position_bias_rate:.1%}  (same position both orders)")
    print(f"  Position flip rate: {bm.position_flip_rate:.1%}  (raw winner changed on swap)")
    print(f"  Score variance:     {bm.score_variance:.3f}  (clustering: {'YES' if bm.score_clustering_detected else 'no'})")
    print(f"  Verbosity probe:    {'PASSED' if bm.verbosity_probe_passed else 'FAILED' if bm.verbosity_probe_passed is not None else 'N/A'}")
    print(f"  Sycophancy probe:   {'PASSED' if bm.sycophancy_probe_passed else 'FAILED' if bm.sycophancy_probe_passed is not None else 'N/A'}")
    print(f"  Self-enhance warn:  {'YES' if bm.self_enhancement_warning else 'No'}")
    print("-" * 60)
    v = report.validation
    print(f"  Validation method:  {v.method}")
    print(f"  Gold agreement:     {v.gold_agreement_rate:.1%}" if v.gold_agreement_rate is not None else "  Gold agreement:     N/A")
    print(f"  Cohen's kappa:      {v.cohens_kappa:.3f}" if v.cohens_kappa is not None else "  Cohen's kappa:      N/A")
    if v.test_retest_agreement is not None:
        print(f"  Test-retest:        {v.test_retest_agreement:.1%}  (n={v.test_retest_sample_size} probes re-judged)")
    else:
        print("  Test-retest:        N/A")
    print(f"  Adversarial probes: {v.adversarial_probes_passed}/{v.adversarial_probes_total}")
    print("-" * 60)
    print(f"  Total judge calls:  {report.total_judge_calls}")
    print(f"  Prompt tokens:      {report.total_prompt_tokens}")
    print(f"  Completion tokens:  {report.total_completion_tokens}")
    print(f"  Total Cost (est):   ${report.total_cost:.4f}")
    print("=" * 60 + "\n")


def _json_safe(report: SuiteReport) -> dict:  # used by tests
    return json.loads(report.model_dump_json())
