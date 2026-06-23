"""Answer-quality judge for faithfulness and relevance on a graded 1-5 scale.

Two backends:

* Real providers (gemini/groq) prompt an LLM that returns a JSON object with
  1-5 scores and rationales, anchored to an explicit rubric. The judge model
  family is kept different from the generator family so the judge is not
  grading its own output.
* The ``mock`` provider is a **deterministic lexical-grounding heuristic** (not
  an LLM). It exists so the evaluation runs offline in CI and, crucially, so a
  planted wrong/unsupported answer is demonstrably scored low without spending
  API quota. Faithfulness is driven by whether the answer's novel claims are
  supported by the retrieved context; relevance by topical overlap with the
  query.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from diskcache import Cache
from groq import Groq

from src.config import settings
from src.gemini_client import call_with_gemini_key
from src.logger import log_event
from src.retry import ROTATABLE_STATUS_CODES, call_with_key_rotation, split_keys, status_code

cache = Cache(settings.cache_path)

SCORE_MIN, SCORE_MAX = 1, 5

# Lexical stopwords for the deterministic mock judge.
_STOP = set(
    "a an and are as at be by for from has have how in into is it its of on or that the to was "
    "what when where which who why with this these those you your we our they their can may will "
    "should would could does do did not no all any more most other some such only same use used "
    "using also based give gives following includes include between mean means".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")


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


def _judge_model_for(provider: str) -> str:
    if settings.judge_model:
        return settings.judge_model
    return settings.groq_model if provider == "groq" else settings.generation_model


def _judge_provider_order() -> list[tuple[str, str]]:
    primary = settings.judge_provider.lower()
    fallback = settings.judge_fallback_provider.lower().strip()
    providers = [primary]
    if fallback and fallback != primary and fallback != "none":
        providers.append(fallback)
    return [(provider, _judge_model_for(provider)) for provider in providers]


def _cache_key(metric: str, prompt: str, provider: str, model: str) -> str:
    return hashlib.sha256(f"judge_v3:{metric}:{provider}:{model}:{prompt}".encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Deterministic mock judge (lexical grounding heuristic)
# --------------------------------------------------------------------------- #
def _content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2]


def _grade(fraction: float) -> int:
    if fraction >= 0.8:
        return 5
    if fraction >= 0.6:
        return 4
    if fraction >= 0.4:
        return 3
    if fraction >= 0.2:
        return 2
    return 1


def _matches(token: str, vocab: set[str]) -> bool:
    """Exact match, or prefix match for longer tokens (open ~ openapi)."""
    if token in vocab:
        return True
    if len(token) >= 4:
        return any(len(v) >= 4 and (v.startswith(token) or token.startswith(v)) for v in vocab)
    return False


def _mock_judge(query: str, context: str, answer: str) -> AnswerJudgeResult:
    answer_tokens = set(_content_tokens(answer))
    context_tokens = set(_content_tokens(context))
    query_tokens = set(_content_tokens(query))

    # Faithfulness: the answer's *novel* claims (tokens not echoing the query)
    # must be supported by the retrieved context.
    novel = [t for t in answer_tokens if t not in query_tokens]
    if novel:
        faith_fraction = sum(1 for t in novel if _matches(t, context_tokens)) / len(novel)
    else:
        faith_fraction = 1.0 if answer_tokens & context_tokens else 0.5

    # Relevance: does the answer address the query's salient terms?
    salient = [t for t in query_tokens if len(t) >= 4] or list(query_tokens)
    if salient:
        rel_fraction = sum(1 for t in salient if _matches(t, answer_tokens)) / len(salient)
    else:
        rel_fraction = 1.0 if answer_tokens else 0.0

    faith = _grade(faith_fraction)
    rel = _grade(rel_fraction)
    raw = json.dumps(
        {
            "faithfulness_score": faith,
            "faithfulness_rationale": f"Mock judge: {faith_fraction:.0%} of the answer's novel claims "
            f"are supported by the retrieved context.",
            "relevance_score": rel,
            "relevance_rationale": f"Mock judge: the answer covers {rel_fraction:.0%} of the query's "
            f"salient terms.",
        }
    )
    return AnswerJudgeResult(
        faithfulness_score=faith,
        faithfulness_rationale=f"Deterministic heuristic: {faith_fraction:.0%} of novel answer claims "
        f"are grounded in the retrieved context.",
        relevance_score=rel,
        relevance_rationale=f"Deterministic heuristic: the answer addresses {rel_fraction:.0%} of the "
        f"query's salient terms.",
        raw_response=raw,
        provider="mock",
        model="lexical-heuristic",
    )


# --------------------------------------------------------------------------- #
# Real LLM judge
# --------------------------------------------------------------------------- #
_RUBRIC = """Score each metric on an integer scale from 1 to 5.

faithfulness_score (is every claim supported by the context?):
  5 = every claim is directly supported by the context and citations are correct
  4 = supported, with at most a trivial unsupported detail
  3 = mostly supported but at least one claim lacks support
  2 = several unsupported or weakly supported claims
  1 = the core claim is unsupported, fabricated, or contradicts the context

relevance_score (does the answer address the user's question?):
  5 = directly and completely answers the question
  4 = answers the question with minor omissions
  3 = partially answers or is somewhat off-target
  2 = mostly off-topic or evasive
  1 = does not address the question at all"""


def _llm_judge_prompt(query: str, context: str, answer: str) -> str:
    return f"""You are a strict RAG evaluator. {_RUBRIC}

Return ONLY one JSON object with exactly these keys:
{{
  "faithfulness_score": <1-5>,
  "faithfulness_rationale": "1-2 concise sentences",
  "relevance_score": <1-5>,
  "relevance_rationale": "1-2 concise sentences"
}}

Question:
{query}

Context:
{context}

Answer:
{answer}
"""


def _call_judge_model(provider: str, model: str, prompt: str) -> str:
    if provider == "gemini":
        response = call_with_gemini_key(
            "judge",
            lambda client: client.models.generate_content(model=model, contents=prompt),
        )
        return response.text or ""
    if provider == "groq":
        keys = split_keys(settings.groq_api_keys)
        if settings.groq_api_key and settings.groq_api_key not in keys:
            keys.append(settings.groq_api_key)
        if not keys:
            raise RuntimeError("GROQ_API_KEY (or GROQ_API_KEYS) is required when JUDGE_PROVIDER=groq.")
        response = call_with_key_rotation(
            keys,
            make_client=lambda key: Groq(api_key=key),
            operation=lambda client: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict RAG evaluator. Output only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            ),
            purpose="judge",
            max_retries=max(1, settings.gemini_max_retries + 1),
        )
        return response.choices[0].message.content or ""
    raise ValueError(f"Unsupported judge provider: {provider}")


def _parse_graded_score(value, field_name: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Judge field {field_name!r} must be an integer {SCORE_MIN}-{SCORE_MAX}.") from exc
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"Judge field {field_name!r} must be {SCORE_MIN}-{SCORE_MAX}, got {score}.")
    return score


def _parse_answer_judge_response(text: str) -> tuple[int, str, int, str]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge did not return a JSON object: {text!r}")

    json_text = re.sub(r",\s*([}\]])", r"\1", raw[start : end + 1])
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {text!r}") from exc

    faithfulness_score = _parse_graded_score(data.get("faithfulness_score"), "faithfulness_score")
    relevance_score = _parse_graded_score(data.get("relevance_score"), "relevance_score")
    faithfulness_rationale = str(data.get("faithfulness_rationale", "")).strip()
    relevance_rationale = str(data.get("relevance_rationale", "")).strip()
    if not faithfulness_rationale or not relevance_rationale:
        raise ValueError(f"Judge JSON must include both rationales: {text!r}")
    return faithfulness_score, faithfulness_rationale, relevance_score, relevance_rationale


def evaluate_answer(query: str, context: str, answer: str) -> AnswerJudgeResult:
    """Judge faithfulness and relevance (1-5) for one answer."""
    provider, model = _judge_provider()

    if not answer or not context:
        return AnswerJudgeResult(
            faithfulness_score=1,
            faithfulness_rationale="Missing answer or context.",
            relevance_score=1,
            relevance_rationale="Missing answer or context.",
            raw_response=json.dumps({"error": "missing answer or context"}),
            provider=provider if provider != "mock" else "mock",
            model=model,
        )

    if provider == "mock":
        return _mock_judge(query, context, answer)

    prompt = _llm_judge_prompt(query, context, answer)
    provider_order = _judge_provider_order()
    last_error: Exception | None = None
    for candidate_provider, candidate_model in provider_order:
        key = _cache_key("answer_judge_v3", prompt, candidate_provider, candidate_model)
        if key in cache:
            return AnswerJudgeResult(**cache[key])
        try:
            raw_response = _call_judge_model(candidate_provider, candidate_model, prompt)
            faith_score, faith_rationale, rel_score, rel_rationale = _parse_answer_judge_response(
                raw_response
            )
            result = AnswerJudgeResult(
                faithfulness_score=faith_score,
                faithfulness_rationale=faith_rationale,
                relevance_score=rel_score,
                relevance_rationale=rel_rationale,
                raw_response=raw_response,
                provider=candidate_provider,
                model=candidate_model,
            )
            cache[key] = asdict(result)
            return result
        except Exception as exc:  # noqa: BLE001 - fallback is based on provider status code
            last_error = exc
            code = status_code(exc)
            can_fallback = (
                code in ROTATABLE_STATUS_CODES and (candidate_provider, candidate_model) != provider_order[-1]
            )
            log_event(
                "judge_provider_failed",
                provider=candidate_provider,
                model=candidate_model,
                status_code=code,
                fallback=can_fallback,
            )
            if can_fallback:
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Judge failed without a captured error.")
