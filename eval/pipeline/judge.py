"""LLM judge: calls the model, extracts JSON, validates with Pydantic, retries on failure."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_FALLBACK_STATUSES = {401, 403, 429, 500, 502, 503, 504}


def _status_code_from(exc: Exception) -> int:
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else 0

from .schemas import JudgeVerdict, CriterionScore, AuditLogEntry
from .config import pipeline_settings
from .prompts import SYSTEM_PROMPT, build_pairwise_prompt, DEFAULT_RUBRIC
from .logger import AuditLogger


COSTS_PER_1M = {
    "gemini-2.5-flash": (0.075, 0.30),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gpt-4o-mini": (0.15, 0.60),
    "mock": (0.0, 0.0),
}


def estimate_cost(model: str, pt: int, ct: int, provider: str | None = None) -> float:
    """Calculate approximate API cost in USD based on token counts.

    Mock runs return $0 regardless of the configured model so the reported cost
    reflects actual API spend, not the fake tokens the mock judge fabricates.
    """
    if provider and provider.lower() == "mock":
        return 0.0
    model_lower = model.lower()
    for prefix, (p_cost, c_cost) in COSTS_PER_1M.items():
        if prefix in model_lower:
            return (pt * p_cost + ct * c_cost) / 1000000.0
    return 0.0


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response, stripping markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    json_str = text[start : end + 1]
    # Fix trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json_str


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse raw LLM response into a validated JudgeVerdict."""
    json_str = _extract_json(raw)
    data = json.loads(json_str)
    return JudgeVerdict.model_validate(data)


def _call_gemini(prompt: str, model: str) -> tuple[str, int, int]:
    """Call Gemini with key rotation and return (text, prompt_tokens, completion_tokens)."""
    from google import genai
    from google.genai import types

    from src.retry import call_with_key_rotation, split_keys

    keys = split_keys(pipeline_settings.gemini_api_keys)
    if pipeline_settings.gemini_api_key and pipeline_settings.gemini_api_key not in keys:
        keys.append(pipeline_settings.gemini_api_key)
    if not keys:
        raise RuntimeError("GEMINI_API_KEY required for judge_provider=gemini")

    # Without an explicit config the SDK defaults to ~1.0, which made the judge
    # non-deterministic across re-runs (test-retest was 0% as a result).
    response = call_with_key_rotation(
        keys,
        make_client=lambda key: genai.Client(api_key=key),
        operation=lambda client: client.models.generate_content(
            model=model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config=types.GenerateContentConfig(
                temperature=pipeline_settings.judge_temperature,
                response_mime_type="application/json",
            ),
        ),
        purpose="judge",
        max_retries=max(1, pipeline_settings.max_judge_retries + 1),
    )
    text = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    pt = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    ct = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    return text, pt, ct


def _groq_keys() -> list[str]:
    from src.retry import split_keys

    keys = split_keys(pipeline_settings.groq_api_keys)
    if pipeline_settings.groq_api_key and pipeline_settings.groq_api_key not in keys:
        keys.append(pipeline_settings.groq_api_key)
    return keys


def _call_groq(prompt: str, model: str) -> tuple[str, int, int]:
    """Call Groq with key rotation and return (text, prompt_tokens, completion_tokens)."""
    from groq import Groq

    from src.retry import call_with_key_rotation

    keys = _groq_keys()
    if not keys:
        raise RuntimeError("GROQ_API_KEY (or GROQ_API_KEYS) required for judge_provider=groq")
    response = call_with_key_rotation(
        keys,
        make_client=lambda key: Groq(api_key=key),
        operation=lambda client: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=pipeline_settings.judge_temperature,
            response_format={"type": "json_object"},
        ),
        purpose="judge",
        max_retries=max(1, pipeline_settings.max_judge_retries + 1),
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return text, pt, ct


def _call_openai(prompt: str, model: str) -> tuple[str, int, int]:
    """Call OpenAI-compatible API (works with OpenRouter too)."""
    import openai
    api_key = pipeline_settings.openai_api_key or pipeline_settings.openrouter_api_key
    base_url = "https://openrouter.ai/api/v1" if pipeline_settings.openrouter_api_key and not pipeline_settings.openai_api_key else None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY required")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=pipeline_settings.judge_temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    return text, pt, ct


def _mock_verdict(output_1: str, output_2: str) -> tuple[str, int, int]:
    """Deterministic mock judge for testing. Prefers shorter, more factual answers."""
    len_ratio = len(output_1) / max(len(output_2), 1)
    winner = "A" if len_ratio <= 1.2 else "B"
    verdict = {
        "criteria_scores_a": [
            {"criterion": "correctness", "score": 4, "rationale": "Mock: appears factual"},
            {"criterion": "completeness", "score": 4, "rationale": "Mock: covers the question"},
            {"criterion": "faithfulness", "score": 4, "rationale": "Mock: grounded"},
            {"criterion": "conciseness", "score": 5 if len(output_1) < len(output_2) else 3, "rationale": "Mock: length-based"},
            {"criterion": "instruction_following", "score": 4, "rationale": "Mock: follows instructions"},
        ],
        "criteria_scores_b": [
            {"criterion": "correctness", "score": 4, "rationale": "Mock: appears factual"},
            {"criterion": "completeness", "score": 4, "rationale": "Mock: covers the question"},
            {"criterion": "faithfulness", "score": 4, "rationale": "Mock: grounded"},
            {"criterion": "conciseness", "score": 5 if len(output_2) < len(output_1) else 3, "rationale": "Mock: length-based"},
            {"criterion": "instruction_following", "score": 4, "rationale": "Mock: follows instructions"},
        ],
        "rationale": f"Mock judge: preferred {'Assistant 1' if winner == 'A' else 'Assistant 2'} based on conciseness.",
        "winner": winner,
    }
    return json.dumps(verdict), 100, 200


def _dispatch_call(prompt: str, output_1: str, output_2: str) -> tuple[str, int, int, str, str]:
    provider = pipeline_settings.judge_provider.lower()
    model = pipeline_settings.judge_model
    fallback_provider = pipeline_settings.judge_fallback_provider.lower().strip()
    fallback_model = pipeline_settings.judge_fallback_model

    provider_order = [(provider, model)]
    if fallback_provider and fallback_provider != "none" and fallback_provider != provider:
        provider_order.append((fallback_provider, fallback_model))

    last_error: Exception | None = None
    for candidate_provider, candidate_model in provider_order:
        try:
            raw, pt, ct = _dispatch_single(candidate_provider, candidate_model, prompt, output_1, output_2)
            return raw, pt, ct, candidate_provider, candidate_model
        except Exception as exc:  # noqa: BLE001 - fallback is based on provider status code
            last_error = exc
            if _status_code_from(exc) in _FALLBACK_STATUSES and (candidate_provider, candidate_model) != provider_order[-1]:
                continue
            raise
    raise last_error if last_error else RuntimeError("judge dispatch failed without a captured error")


def _dispatch_single(provider: str, model: str, prompt: str, output_1: str, output_2: str) -> tuple[str, int, int]:
    if provider == "mock":
        return _mock_verdict(output_1, output_2)
    if provider == "gemini":
        return _call_gemini(prompt, model)
    if provider == "groq":
        return _call_groq(prompt, model)
    if provider in ("openai", "openrouter"):
        return _call_openai(prompt, model)
    raise ValueError(f"Unsupported judge provider: {provider}")


def call_judge(
    case_id: str,
    user_input: str,
    output_1: str,
    output_2: str,
    order: str,
    audit_logger: AuditLogger,
    context: str | None = None,
    reference: str | None = None,
) -> JudgeVerdict:
    """Call the LLM judge with retry logic and full audit logging."""
    prompt = build_pairwise_prompt(
        user_input=user_input,
        output_1=output_1,
        output_2=output_2,
        context=context,
        reference=reference,
    )
    last_error: Exception | None = None
    raw = ""
    actual_provider = pipeline_settings.judge_provider
    actual_model = pipeline_settings.judge_model
    for attempt in range(1 + pipeline_settings.max_judge_retries):
        start = time.time()
        try:
            raw, pt, ct, actual_provider, actual_model = _dispatch_call(prompt, output_1, output_2)
            latency = (time.time() - start) * 1000
            verdict = _parse_verdict(raw)
            cost = estimate_cost(
                actual_model, pt, ct, provider=actual_provider
            )
            audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                case_id=case_id,
                order=order,
                judge_provider=actual_provider,
                judge_model=actual_model,
                prompt=prompt,
                raw_response=raw,
                parsed_verdict=verdict.model_dump(),
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                cost_estimate=cost,
                latency_ms=round(latency, 2),
                retry_count=attempt,
            ))
            return verdict
        except Exception as exc:
            latency = (time.time() - start) * 1000
            last_error = exc
            audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                case_id=case_id,
                order=order,
                judge_provider=actual_provider,
                judge_model=actual_model,
                prompt=prompt,
                raw_response=raw,
                parse_error=f"{type(exc).__name__}: {exc}",
                latency_ms=round(latency, 2),
                retry_count=attempt,
            ))
            if attempt < pipeline_settings.max_judge_retries:
                # Real backoff for transient provider errors (429/5xx). Without
                # this, a brief Gemini "503 UNAVAILABLE" spike turns into 100%
                # error rate because all 3 attempts fire within milliseconds.
                code = _status_code_from(exc)
                if code in _RETRYABLE_STATUSES:
                    time.sleep(min(2 ** attempt, 8))
                else:
                    # Likely a parse failure; nudge the model and retry immediately.
                    prompt += f"\n\nYour previous response was invalid. Error: {exc}. Please output ONLY valid JSON."
                continue

    # All retries exhausted: return a safe fallback Error verdict
    fallback_scores = [
        CriterionScore(criterion=c.name, score=0, rationale="Parse failure fallback")
        for c in DEFAULT_RUBRIC.criteria
    ]
    return JudgeVerdict(
        criteria_scores_a=fallback_scores,
        criteria_scores_b=list(fallback_scores),
        rationale=f"Judge output could not be parsed after {pipeline_settings.max_judge_retries + 1} attempts: {last_error}",
        winner="Error",
    )
