"""Pairwise evaluation with position-swap debiasing."""
from __future__ import annotations

from .schemas import TestCase, JudgeVerdict, PairwiseResult
from .judge import call_judge
from .logger import AuditLogger


def _resolve_winner(
    verdict_ab: JudgeVerdict,
    verdict_ba: JudgeVerdict,
) -> tuple[str, bool]:
    """Resolve final winner from two orderings. Returns (winner, position_bias_detected).

    Logic:
    - verdict_ab.winner is from the perspective of (A=Asst1, B=Asst2)
      so "A" means A won, "B" means B won.
    - verdict_ba.winner is from the perspective of (B=Asst1, A=Asst2)
      so "A" means Asst1 won = B won, "B" means Asst2 won = A won.

    Agreement table:
      ab=A, ba=B -> both say A wins (ba's "B" = Asst2 = original A) -> A wins
      ab=B, ba=A -> both say B wins (ba's "A" = Asst1 = original B) -> B wins
      ab=A, ba=A -> both pick Asst1 regardless -> POSITION BIAS
      ab=B, ba=B -> both pick Asst2 regardless -> POSITION BIAS (position-2 preference)
      ab=Tie, ba=Tie -> Tie
      mixed with Tie -> lean toward the non-Tie verdict
    """
    w_ab = verdict_ab.winner  # A, B, Tie, or Error from (A=1, B=2) perspective
    w_ba = verdict_ba.winner  # A, B, Tie, or Error from (B=1, A=2) perspective

    if w_ab == "Error" or w_ba == "Error":
        return "Error", False

    # Map ba verdict back to original labels:
    # ba's "A" (Asst1) = original B; ba's "B" (Asst2) = original A
    ba_mapped = {"A": "B", "B": "A", "Tie": "Tie"}[w_ba]

    if w_ab == ba_mapped:
        # Both orders agree on the same original answer
        return w_ab, False

    # Check for position bias: judge always picks the same position
    if w_ab == "A" and w_ba == "A":
        # Both times picked Assistant 1 -> position-1 bias
        return "Conflict", True
    if w_ab == "B" and w_ba == "B":
        # Both times picked Assistant 2 -> position-2 bias
        return "Conflict", True

    # One is Tie, one is not -> lean toward the decisive verdict
    if w_ab == "Tie" and ba_mapped != "Tie":
        return ba_mapped, False
    if ba_mapped == "Tie" and w_ab != "Tie":
        return w_ab, False

    # Disagreement (e.g., ab=A, ba_mapped=B) -> Conflict
    return "Conflict", True


def evaluate_case(
    case: TestCase,
    audit_logger: AuditLogger,
) -> PairwiseResult:
    """Evaluate one test case in both A/B orders and resolve the winner."""
    # Run 1: A as Assistant 1, B as Assistant 2
    verdict_ab = call_judge(
        case_id=case.id,
        user_input=case.input,
        output_1=case.output_a,
        output_2=case.output_b,
        order="AB",
        audit_logger=audit_logger,
        context=case.context,
        reference=case.reference_output,
    )

    # Run 2: B as Assistant 1, A as Assistant 2
    verdict_ba = call_judge(
        case_id=case.id,
        user_input=case.input,
        output_1=case.output_b,
        output_2=case.output_a,
        order="BA",
        audit_logger=audit_logger,
        context=case.context,
        reference=case.reference_output,
    )

    final_winner, position_bias = _resolve_winner(verdict_ab, verdict_ba)

    return PairwiseResult(
        case_id=case.id,
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
        final_winner=final_winner,
        position_bias_detected=position_bias,
        gold_winner=case.gold_winner,
    )
