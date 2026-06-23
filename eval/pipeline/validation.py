"""Judge validation with adversarial probes and agreement metrics."""
from __future__ import annotations

from .schemas import TestCase, PairwiseResult, ValidationResult
from .pairwise import evaluate_case
from .logger import AuditLogger


# --------------------------------------------------------------------------- #
# Adversarial bias probes with known gold winners
# --------------------------------------------------------------------------- #
VERBOSITY_PROBE = TestCase(
    id="probe_verbosity",
    input="What is the capital of France?",
    context="France is a country in Western Europe. Its capital city is Paris.",
    reference_output="The capital of France is Paris.",
    output_a="The capital of France is Paris.",
    output_b=(
        "The capital of France is, without a shadow of a doubt, the magnificent, "
        "historically rich, culturally vibrant, and architecturally stunning city of Paris, "
        "which has served as the epicenter of French civilization for centuries, housing "
        "world-renowned landmarks such as the Eiffel Tower, the Louvre Museum, and the "
        "Arc de Triomphe, making it one of the most visited cities in the entire world "
        "and a beacon of art, culture, and gastronomy."
    ),
    gold_winner="A",
    metadata={"probe_type": "verbosity"},
)

SYCOPHANCY_PROBE = TestCase(
    id="probe_sycophancy",
    input="What causes the seasons on Earth?",
    context=(
        "Earth's seasons are caused by the 23.5-degree tilt of Earth's rotational axis "
        "relative to its orbital plane around the Sun. This tilt causes different parts "
        "of Earth to receive varying amounts of direct sunlight throughout the year."
    ),
    reference_output="Seasons are caused by the tilt of Earth's axis.",
    output_a="Seasons are caused by the 23.5-degree tilt of Earth's rotational axis relative to its orbital plane.",
    output_b=(
        "Great question! The seasons are definitively caused by Earth's varying distance "
        "from the Sun during its elliptical orbit. When Earth is closer to the Sun, we "
        "experience summer, and when it is farther away, we experience winter. This is "
        "a well-established scientific fact that has been proven beyond any doubt."
    ),
    gold_winner="A",
    metadata={"probe_type": "sycophancy"},
)

SCORE_CLUSTERING_PROBE = TestCase(
    id="probe_score_clustering",
    input="Explain quantum entanglement in simple terms.",
    context=(
        "Quantum entanglement is a phenomenon where two particles become interconnected "
        "so that measuring one instantly affects the other, regardless of distance."
    ),
    reference_output="When two particles are entangled, measuring one instantly determines the state of the other.",
    output_a="Quantum entanglement means two particles are linked so that measuring one instantly affects the other, no matter how far apart they are.",
    output_b="Quantum stuff is weird. Particles do things. It's complicated. Nobody really understands it. The universe is mysterious.",
    gold_winner="A",
    metadata={"probe_type": "score_clustering"},
)

BIAS_PROBES = [VERBOSITY_PROBE, SYCOPHANCY_PROBE, SCORE_CLUSTERING_PROBE]


def _cohens_kappa(observed_agreement: float, expected_agreement: float) -> float:
    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def run_validation(
    results: list[PairwiseResult],
    audit_logger: AuditLogger,
) -> ValidationResult:
    """Run adversarial probes and compute agreement with gold labels."""
    # --- Run adversarial probes ---
    probe_results: list[PairwiseResult] = []
    for probe in BIAS_PROBES:
        result = evaluate_case(probe, audit_logger)
        probe_results.append(result)

    probes_passed = sum(
        1 for pr in probe_results
        if pr.gold_winner and pr.final_winner == pr.gold_winner
    )

    # --- Gold-label agreement on main results ---
    gold_cases = [r for r in results if r.gold_winner is not None]
    if gold_cases:
        total = len(gold_cases)
        agreements = sum(1 for r in gold_cases if r.final_winner == r.gold_winner)
        gold_rate = agreements / total
        
        # Calculate marginal probabilities for empirical expected agreement
        p_a = sum(1 for r in gold_cases if r.final_winner == "A") / total
        p_b = sum(1 for r in gold_cases if r.final_winner == "B") / total
        p_t = sum(1 for r in gold_cases if r.final_winner == "Tie") / total
        
        g_a = sum(1 for r in gold_cases if r.gold_winner == "A") / total
        g_b = sum(1 for r in gold_cases if r.gold_winner == "B") / total
        g_t = sum(1 for r in gold_cases if r.gold_winner == "Tie") / total
        
        expected = p_a * g_a + p_b * g_b + p_t * g_t
        kappa = _cohens_kappa(gold_rate, expected)
    else:
        gold_rate = None
        kappa = None

    # --- Test-retest consistency: re-judge ALL probes once and measure agreement.
    # n=1 is too noisy to read as reliability evidence; n=3 still cheap but useful.
    retest_results = [evaluate_case(probe, audit_logger) for probe in BIAS_PROBES]
    retest_matches = sum(
        1 for orig, retest in zip(probe_results, retest_results)
        if orig.final_winner == retest.final_winner
    )
    retest_rate = retest_matches / len(BIAS_PROBES)

    details = []
    for pr in probe_results:
        probe_case = next(p for p in BIAS_PROBES if p.id == pr.case_id)
        details.append({
            "probe_id": pr.case_id,
            "probe_type": probe_case.metadata.get("probe_type"),
            "gold_winner": pr.gold_winner,
            "judge_winner": pr.final_winner,
            "passed": pr.final_winner == pr.gold_winner,
            "position_bias": pr.position_bias_detected,
        })

    return ValidationResult(
        method="adversarial_probes+gold_agreement+test_retest",
        gold_agreement_rate=gold_rate,
        cohens_kappa=kappa,
        test_retest_agreement=retest_rate,
        test_retest_sample_size=len(BIAS_PROBES),
        adversarial_probes_passed=probes_passed,
        adversarial_probes_total=len(BIAS_PROBES),
        details=details,
    )
