import hashlib
import json
import re
from dataclasses import asdict, dataclass

from diskcache import Cache
from groq import Groq

from src.config import settings
from src.gemini_client import call_with_gemini_key

cache = Cache(settings.cache_path)
SCORE_RE = re.compile(r"^SCORE\s*:\s*([01])\s*$", re.IGNORECASE)


@dataclass
class JudgeResult:
    score: int
    rationale: str
    raw_response: str
    provider: str
    model: str


@dataclass
class AnswerJudgeResult:
    faithfulness_score: int
    faithfulness_rationale: str
    relevance_score: int
    relevance_rationale: str
    raw_response: str
    provider: str
    model: str


def _judge_provider() -> tuple[str, str]:
    provider = settings.judge_provider.lower()
    default_model = settings.groq_model if provider == "groq" else settings.generation_model
    model = settings.judge_model or default_model
    return provider, model


def _cache_key(metric: str, prompt: str, provider: str, model: str) -> str:
    return hashlib.sha256(f"judge_v2:{metric}:{provider}:{model}:{prompt}".encode()).hexdigest()


def _parse_judge_response(text: str) -> tuple[int, str]:
    raw = (text or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""
    match = SCORE_RE.match(final_line)
    if not match:
        raise ValueError(f"Judge did not end with 'SCORE: 0' or 'SCORE: 1': {text!r}")

    rationale = "\n".join(lines[:-1]).strip()
    return int(match.group(1)), rationale


def _parse_binary_score(value, field_name: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Judge field {field_name!r} must be 0 or 1.") from exc
    if score not in (0, 1):
        raise ValueError(f"Judge field {field_name!r} must be 0 or 1.")
    return score


def _parse_answer_judge_response(text: str) -> tuple[int, str, int, str]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge did not return a JSON object: {text!r}")

    json_text = raw[start : end + 1]
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {text!r}") from exc

    faithfulness_score = _parse_binary_score(data.get("faithfulness_score"), "faithfulness_score")
    relevance_score = _parse_binary_score(data.get("relevance_score"), "relevance_score")
    faithfulness_rationale = str(data.get("faithfulness_rationale", "")).strip()
    relevance_rationale = str(data.get("relevance_rationale", "")).strip()

    if not faithfulness_rationale or not relevance_rationale:
        raise ValueError(f"Judge JSON must include both rationales: {text!r}")

    return faithfulness_score, faithfulness_rationale, relevance_score, relevance_rationale


def _call_judge_model(provider: str, prompt: str) -> str:
    if provider == "mock":
        return "Rationale: Mock judge accepts the answer for offline testing.\nSCORE: 1"
    if provider == "gemini":
        response = call_with_gemini_key(
            "judge",
            lambda client: client.models.generate_content(
                model=settings.judge_model or settings.generation_model,
                contents=prompt,
            ),
        )
        return response.text or ""
    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when JUDGE_PROVIDER=groq.")
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.judge_model or settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluator. Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"Unsupported judge provider: {settings.judge_provider}")


def _run_judge(metric: str, prompt: str) -> JudgeResult:
    provider, model = _judge_provider()
    key = _cache_key(metric, prompt, provider, model)
    if key in cache:
        return JudgeResult(**cache[key])

    raw_response = _call_judge_model(provider, prompt)

    score, rationale = _parse_judge_response(raw_response)
    result = JudgeResult(
        score=score,
        rationale=rationale,
        raw_response=raw_response,
        provider=provider,
        model=model,
    )
    cache[key] = asdict(result)
    return result


def evaluate_answer_detail(query: str, context: str, answer: str) -> AnswerJudgeResult:
    """Returns one cost-efficient judge result for faithfulness and relevance."""
    if not answer or not context:
        return AnswerJudgeResult(
            faithfulness_score=0,
            faithfulness_rationale="Missing answer or context.",
            relevance_score=0,
            relevance_rationale="Missing answer or context.",
            raw_response=json.dumps(
                {
                    "faithfulness_score": 0,
                    "faithfulness_rationale": "Missing answer or context.",
                    "relevance_score": 0,
                    "relevance_rationale": "Missing answer or context.",
                }
            ),
            provider="local",
            model="rule",
        )

    provider, model = _judge_provider()
    prompt = f"""Evaluate this RAG answer for two metrics in one pass.

Metric definitions:
- faithfulness_score: 1 only if every factual claim in the answer is directly supported by the provided context and the cited chunk IDs support the claim; otherwise 0.
- relevance_score: 1 only if the answer directly addresses the user's query; otherwise 0.

Return only one JSON object with exactly these keys:
{{
  "faithfulness_score": 0 or 1,
  "faithfulness_rationale": "1-2 concise sentences",
  "relevance_score": 0 or 1,
  "relevance_rationale": "1-2 concise sentences"
}}

Query:
{query}

Context:
{context}

Answer:
{answer}
"""
    key = _cache_key("answer_judge_v1", prompt, provider, model)
    if key in cache:
        return AnswerJudgeResult(**cache[key])

    if provider == "mock":
        raw_response = json.dumps(
            {
                "faithfulness_score": 1,
                "faithfulness_rationale": "Mock judge accepts the groundedness claim for offline testing.",
                "relevance_score": 1,
                "relevance_rationale": "Mock judge accepts the relevance claim for offline testing.",
            }
        )
    else:
        raw_response = _call_judge_model(provider, prompt)

    faithfulness_score, faithfulness_rationale, relevance_score, relevance_rationale = (
        _parse_answer_judge_response(raw_response)
    )
    result = AnswerJudgeResult(
        faithfulness_score=faithfulness_score,
        faithfulness_rationale=faithfulness_rationale,
        relevance_score=relevance_score,
        relevance_rationale=relevance_rationale,
        raw_response=raw_response,
        provider=provider,
        model=model,
    )
    cache[key] = asdict(result)
    return result


def evaluate_faithfulness_detail(context: str, answer: str) -> JudgeResult:
    """Returns a judge score and rationale for whether the answer is grounded."""
    if not answer or not context:
        return JudgeResult(
            score=0,
            rationale="Missing answer or context.",
            raw_response="Missing answer or context.\nSCORE: 0",
            provider="local",
            model="rule",
        )

    prompt = f"""Evaluate faithfulness for a RAG answer.

Criteria:
- SCORE: 1 only if every factual claim in the answer is directly supported by the provided context.
- SCORE: 0 if the answer contains unsupported claims, contradictions, or citations that do not support the claim.

Write 1-2 concise sentences explaining your reasoning.
End with exactly one final line: SCORE: 0 or SCORE: 1

Context:
{context}

Answer:
{answer}
"""
    return _run_judge("faithfulness", prompt)


def evaluate_relevance_detail(query: str, answer: str) -> JudgeResult:
    """Returns a judge score and rationale for whether the answer addresses the query."""
    if not answer:
        return JudgeResult(
            score=0,
            rationale="Missing answer.",
            raw_response="Missing answer.\nSCORE: 0",
            provider="local",
            model="rule",
        )

    prompt = f"""Evaluate answer relevance.

Criteria:
- SCORE: 1 only if the answer directly addresses the user's query.
- SCORE: 0 if the answer is off-topic, evasive, or mainly answers a different question.

Write 1-2 concise sentences explaining your reasoning.
End with exactly one final line: SCORE: 0 or SCORE: 1

Query:
{query}

Answer:
{answer}
"""
    return _run_judge("relevance", prompt)


def evaluate_faithfulness(context: str, answer: str) -> int:
    return evaluate_faithfulness_detail(context, answer).score


def evaluate_relevance(query: str, answer: str) -> int:
    return evaluate_relevance_detail(query, answer).score
