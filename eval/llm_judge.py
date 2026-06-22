import hashlib
from groq import Groq
from diskcache import Cache
from src.config import settings
from src.gemini_client import call_with_gemini_key

cache = Cache(settings.cache_path)

def _judge_provider() -> tuple[str, str]:
    provider = settings.judge_provider.lower()
    model = settings.groq_model if provider == "groq" else settings.generation_model
    return provider, model

def _cache_key(metric: str, prompt: str, provider: str, model: str) -> str:
    return hashlib.sha256(f"judge:{metric}:{provider}:{model}:{prompt}".encode()).hexdigest()

def _parse_binary_score(text: str) -> int:
    cleaned = (text or "").strip()
    if cleaned.startswith("1"):
        return 1
    if cleaned.startswith("0"):
        return 0
    raise ValueError(f"Judge did not return a binary score: {text!r}")

def _run_judge(metric: str, prompt: str) -> int:
    provider, model = _judge_provider()
    key = _cache_key(metric, prompt, provider, model)
    if key in cache:
        return int(cache[key])

    if provider == "mock":
        score = 1
    elif provider == "gemini":
        response = call_with_gemini_key(
            "judge",
            lambda client: client.models.generate_content(
                model=settings.generation_model,
                contents=prompt,
            ),
        )
        score = _parse_binary_score(response.text)
    elif provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when JUDGE_PROVIDER=groq.")
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Return only 1 or 0."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        score = _parse_binary_score(response.choices[0].message.content)
    else:
        raise ValueError(f"Unsupported judge provider: {settings.judge_provider}")

    cache[key] = score
    return score

def evaluate_faithfulness(context: str, answer: str) -> int:
    """Returns 1 if the answer is completely supported by the context, else 0."""
    if not answer or not context:
        return 0

    prompt = f"""Given the context and answer, evaluate faithfulness.
Return only 1 if every factual statement in the answer is supported by the context.
Return only 0 if the answer contains unsupported claims, contradictions, or fabricated facts.

Context:
{context}

Answer:
{answer}

Score:"""
    return _run_judge("faithfulness", prompt)

def evaluate_relevance(query: str, answer: str) -> int:
    """Returns 1 if the answer directly addresses the query, else 0."""
    if not answer:
        return 0
        
    prompt = f"""Given the following query and answer, evaluate whether the answer directly and correctly addresses the query.
Output only '1' if the answer is relevant and directly addresses the query, or '0' if it is irrelevant or evasive.

Query:
{query}

Answer:
{answer}

Relevance Score (1 or 0):"""

    return _run_judge("relevance", prompt)
