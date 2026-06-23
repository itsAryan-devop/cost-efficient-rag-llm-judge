from __future__ import annotations
import enum
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation case from a test suite."""
    id: str
    input: str = Field(description="The user query or instruction")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")
    context: str | None = Field(default=None, description="Retrieved context for RAG")
    reference_output: str | None = Field(default=None, description="Gold/expected answer")
    output_a: str = Field(description="Response from config A")
    output_b: str = Field(description="Response from config B")
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold_winner: Literal["A", "B", "Tie"] | None = Field(default=None, description="Human gold label for validation")


class TestSuite(BaseModel):
    """Full test suite loaded from JSON/YAML."""
    name: str = "default"
    description: str = ""
    config_a: str = Field(default="Config A", description="Label for configuration A")
    config_b: str = Field(default="Config B", description="Label for configuration B")
    cases: list[TestCase]


class RubricCriterion(BaseModel):
    """One criterion in the evaluation rubric."""
    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)


class Rubric(BaseModel):
    """Explicit rubric with named criteria and anchored scale."""
    criteria: list[RubricCriterion]
    scale_min: int = 1
    scale_max: int = 5


class CriterionScore(BaseModel):
    """Score for one criterion."""
    criterion: str
    score: int
    rationale: str


class JudgeVerdict(BaseModel):
    """Structured verdict from the LLM judge for one pairwise comparison."""
    criteria_scores_a: list[CriterionScore] = Field(description="Per-criterion scores for Assistant 1")
    criteria_scores_b: list[CriterionScore] = Field(description="Per-criterion scores for Assistant 2")
    rationale: str = Field(description="Overall reasoning before declaring winner")
    winner: Literal["A", "B", "Tie", "Error"] = Field(description="Which assistant is better overall")


class PairwiseResult(BaseModel):
    """Result of one test case evaluated in both A/B orders."""
    case_id: str
    verdict_ab: JudgeVerdict = Field(description="Verdict with A as Assistant 1, B as Assistant 2")
    verdict_ba: JudgeVerdict = Field(description="Verdict with B as Assistant 1, A as Assistant 2")
    final_winner: Literal["A", "B", "Tie", "Conflict", "Error"] = Field(description="Resolved winner after debiasing")
    position_bias_detected: bool = Field(default=False)
    gold_winner: Literal["A", "B", "Tie"] | None = Field(default=None)


class BiasMetrics(BaseModel):
    """Aggregate bias measurements across the suite."""
    position_bias_rate: float = Field(
        description="Fraction of cases where judge picked the same position in BOTH orders (always Assistant 1, or always Assistant 2)."
    )
    position_flip_rate: float = Field(
        description="Fraction of cases where the raw winner CHANGED between the AB and BA orderings, measured independently of how we resolved the case."
    )
    verbosity_probe_passed: bool | None = Field(default=None, description="True if judge correctly penalized verbose fluff")
    sycophancy_probe_passed: bool | None = Field(default=None, description="True if judge correctly penalized confident-but-wrong")
    self_enhancement_warning: bool = Field(default=False, description="True if judge and generator are same model family")
    score_variance: float = Field(
        default=0.0,
        description="Variance of criterion scores across the suite. Low values indicate the judge is bunching scores in a narrow band (score clustering)."
    )
    score_clustering_detected: bool = Field(
        default=False,
        description="True when the standard deviation of criterion scores is below 0.5 (judge is not discriminating across the 1-5 scale)."
    )


class ValidationResult(BaseModel):
    """Judge validation evidence."""
    method: str
    gold_agreement_rate: float | None = Field(default=None)
    cohens_kappa: float | None = Field(default=None)
    test_retest_agreement: float | None = Field(default=None)
    test_retest_sample_size: int = Field(
        default=0,
        description="Number of probes re-judged for test-retest. Small (e.g. n=1) means this is a smoke check, not a reliability estimate.",
    )
    adversarial_probes_passed: int = 0
    adversarial_probes_total: int = 0
    details: list[dict[str, Any]] = Field(default_factory=list)


class CriterionAggregate(BaseModel):
    """Aggregate scores for one criterion across the suite (per assistant)."""
    criterion: str
    weight: float
    mean_score_a: float
    mean_score_b: float
    cases_scored: int


class SuiteReport(BaseModel):
    """Final suite-level evaluation report."""
    suite_name: str
    created_at: str
    config_a: str
    config_b: str
    total_cases: int
    win_rate_a: float
    win_rate_b: float
    tie_rate: float
    conflict_rate: float
    error_rate: float
    winner: str = Field(description="Declared suite-level winner or Tie")
    # ---- Rubric-driven aggregates (rubric weights are now actually used) ----
    criterion_aggregates: list[CriterionAggregate] = Field(
        default_factory=list,
        description="Per-criterion mean scores for A and B, averaged across cases (both AB and BA orderings).",
    )
    weighted_score_a: float = Field(
        default=0.0,
        description="Suite-level weighted mean score for A: sum(mean * weight) / sum(weight) over criteria.",
    )
    weighted_score_b: float = Field(
        default=0.0,
        description="Suite-level weighted mean score for B (same formula).",
    )
    pass_threshold: float = Field(
        default=4.0,
        description="Per-case weighted-score cutoff used for the pass rate (>=).",
    )
    pass_rate_a: float = Field(
        default=0.0,
        description="Fraction of cases whose weighted score for A is >= pass_threshold.",
    )
    pass_rate_b: float = Field(
        default=0.0,
        description="Fraction of cases whose weighted score for B is >= pass_threshold.",
    )
    bias_metrics: BiasMetrics
    validation: ValidationResult
    judge_provider: str
    judge_model: str
    generator_provider: str
    generator_model: str
    total_judge_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    results: list[PairwiseResult]


class AuditLogEntry(BaseModel):
    """One logged judge API call for full auditability."""
    timestamp: str
    case_id: str
    order: Literal["AB", "BA"]
    judge_provider: str
    judge_model: str
    prompt: str
    raw_response: str
    parsed_verdict: dict[str, Any] | None = None
    parse_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    latency_ms: float = 0.0
    retry_count: int = 0
