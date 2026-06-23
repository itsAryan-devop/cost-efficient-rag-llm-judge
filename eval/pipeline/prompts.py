from .schemas import Rubric, RubricCriterion


DEFAULT_RUBRIC = Rubric(
    criteria=[
        RubricCriterion(
            name="correctness",
            description="Factual accuracy and correctness of the response. Are claims supported by the context or verifiable?",
            weight=2.0,
        ),
        RubricCriterion(
            name="completeness",
            description="Does the response fully address the question? Are important aspects covered?",
            weight=1.5,
        ),
        RubricCriterion(
            name="faithfulness",
            description="Is the response grounded in the provided context? Does it avoid hallucination or unsupported claims?",
            weight=2.0,
        ),
        RubricCriterion(
            name="conciseness",
            description="Is the response appropriately concise? Penalize unnecessary padding, filler, or verbose fluff that adds no information.",
            weight=1.0,
        ),
        RubricCriterion(
            name="instruction_following",
            description="Does the response follow the instructions and constraints given in the prompt?",
            weight=1.5,
        ),
    ],
    scale_min=1,
    scale_max=5,
)


SYSTEM_PROMPT = """You are an impartial, rigorous AI evaluation judge. Your task is to compare two AI assistant responses and determine which one is better.

CRITICAL RULES:
- Judge ONLY on substance and factual quality, never on style, politeness, or confidence.
- A confident-sounding but factually wrong answer MUST score lower than a hesitant but correct one.
- A longer answer is NOT better unless the extra content is substantive and supported.
- Penalize unsupported claims, hallucinations, and filler regardless of how fluent they sound.
- Evaluate each criterion independently before deciding the overall winner.
- You MUST output valid JSON and nothing else."""


def _format_rubric(rubric: Rubric) -> str:
    lines = [f"Score each criterion on a {rubric.scale_min}-{rubric.scale_max} scale:"]
    for c in rubric.criteria:
        lines.append(f"")
        lines.append(f"  {c.name} (weight {c.weight}):")
        lines.append(f"    {c.description}")
        lines.append(f"    {rubric.scale_min} = completely fails this criterion")
        mid = (rubric.scale_min + rubric.scale_max) // 2
        lines.append(f"    {mid} = partially meets this criterion")
        lines.append(f"    {rubric.scale_max} = fully meets this criterion")
    return "\n".join(lines)


def build_pairwise_prompt(
    user_input: str,
    output_1: str,
    output_2: str,
    rubric: Rubric | None = None,
    context: str | None = None,
    reference: str | None = None,
    max_chars: int = 50000,
) -> str:
    rubric = rubric or DEFAULT_RUBRIC
    rubric_text = _format_rubric(rubric)

    if len(output_1) > max_chars:
        output_1 = output_1[:max_chars] + "\n\n...[TRUNCATED TO PREVENT CONTEXT OVERFLOW]"
    if len(output_2) > max_chars:
        output_2 = output_2[:max_chars] + "\n\n...[TRUNCATED TO PREVENT CONTEXT OVERFLOW]"

    criteria_json_a = ", ".join(
        f'{{"criterion": "{c.name}", "score": <{rubric.scale_min}-{rubric.scale_max}>, "rationale": "..."}}'
        for c in rubric.criteria
    )
    criteria_json_b = criteria_json_a

    parts = [
        "Compare the two assistant responses below.",
        "",
        f"=== USER INPUT ===",
        user_input,
    ]

    if context:
        parts += ["", "=== CONTEXT ===", context]
    if reference:
        parts += ["", "=== REFERENCE ANSWER ===", reference]

    parts += [
        "",
        "=== ASSISTANT 1 ===",
        output_1,
        "",
        "=== ASSISTANT 2 ===",
        output_2,
        "",
        "=== RUBRIC ===",
        rubric_text,
        "",
        "=== INSTRUCTIONS ===",
        "1. Score each criterion for BOTH assistants.",
        "2. Write your overall reasoning comparing the two.",
        "3. Declare the winner: \"A\" (Assistant 1), \"B\" (Assistant 2), or \"Tie\".",
        "4. Output ONLY the following JSON object, no other text:",
        "",
        "```json",
        "{",
        f'  \"criteria_scores_a\": [{criteria_json_a}],',
        f'  \"criteria_scores_b\": [{criteria_json_b}],',
        '  \"rationale\": \"Your overall reasoning here\",',
        '  \"winner\": \"A\" or \"B\" or \"Tie\"',
        "}",
        "```",
    ]
    return "\n".join(parts)
