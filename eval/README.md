# Problem 2: LLM-as-Judge Evaluation Pipeline

This directory contains a production-grade, bias-aware evaluation pipeline for RAG and LLM systems. It solves the core problems of LLM-as-Judge evaluations by treating the judge not as an infallible oracle, but as a system that requires strict schemas, debiasing, and empirical validation.

## Pipeline Architecture
The pipeline is divided into independent modules:
*   `schemas.py`: Strict data contracts using Pydantic.
*   `prompts.py`: Explicit rubrics (1-5 scale) and structure forcing.
*   `judge.py`: Robust LLM execution. It extracts JSON from Markdown fences, maps to Pydantic, and performs automatic retries if the judge outputs invalid JSON.
*   `pairwise.py`: The debiasing engine. Orchestrates A/B comparisons with double-blind position swapping.
*   `validation.py`: Validation probes to measure human agreement, test-retest consistency, and bias detection.
*   `report.py`: Aggregation and metric calculation.

## How to Run

1.  Set your API keys in `.env` (e.g., `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).
2.  Set the `JUDGE_PROVIDER` in `.env`.
3.  Execute the pipeline from the project root:

```bash
# Ensure PYTHONPATH is set to the project root
python -m eval.run_eval --suite eval/suites/sample_suite.yaml
```

The system will generate:
1.  **Console Report**: A human-readable summary of win rates and biases.
2.  **Audit Log (`reports/p2_audit_log.jsonl`)**: Every prompt, raw response, tokens used, and latency for 100% auditability.
3.  **JSON Report (`reports/p2_evaluation_report.json`)**: Machine-readable aggregated results and validation details.
4.  **CSV Summary (`reports/p2_results.csv`)**: Row-level outcomes for quick spreadsheet analysis.

## Discussion: Bias and Mitigation

Evaluating LLMs with LLMs introduces significant biases. This pipeline addresses them explicitly:

### 1. Position / A-B Order Bias
*   **Before Mitigation**: LLM judges frequently suffer from "primacy bias," strongly preferring the first answer they read (Assistant 1), even if it's identical or worse than Assistant 2.
*   **After Mitigation**: We enforce **Pairwise Double-Blind Swapping**. Every comparison is run twice: `(A, B)` and `(B, A)`. If the judge picks Assistant 1 both times, we detect the position bias and mark the result as a `Conflict` rather than awarding a false win.
*   **Metric tracked**: `position_bias_rate` and `position_flip_rate`.

### 2. Verbosity & Sycophancy Biases
*   **Detection**: The pipeline runs an adversarial probe suite (`validation.py`). It injects a "Verbosity Probe" (a highly padded answer against a concise correct one) and a "Sycophancy Probe" (an extremely polite, confident, but factually incorrect answer).
*   **Mitigation**: We mitigate this through explicit prompting in the rubric (penalizing fluff) and through chain-of-thought grounding (forcing the rationale to be output before the final score).

### 3. Score Clustering
*   Pointwise grading often results in all answers receiving 4s and 5s. By making Pairwise comparison our default mode, we force the LLM to differentiate and declare a winner, breaking the clustering effect.

### 4. Self-Enhancement Bias
*   The system configurations independently track the `JUDGE_PROVIDER` and `GENERATOR_PROVIDER`. The pipeline explicitly warns (`self_enhancement_warning`) if the judge is grading its own model family.

## Conclusion: Is this good enough to gate a release?

**Yes.**
A naive LLM-as-Judge setup (just asking GPT-4 to "score this 1-10") is mathematically unsafe for gating a release due to high position bias and variance. 

However, this implementation **is good enough** because it acts defensively:
1.  **Validation-Gated**: We know if we can trust the judge because we run adversarial probes. If the judge fails the sycophancy probe, we know the run is invalid.
2.  **Statistically Sound**: By running the double-blind `(A, B)` and `(B, A)` swap, we eliminate false wins caused by ordering bias. If a configuration wins under this system, it won by merit, not by luck of position.
3.  **Auditable**: Because every API call is logged with its raw response, engineers can spot-check the exact chain-of-thought rationale for any disputed verdict.
