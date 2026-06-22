import hashlib
from dataclasses import dataclass

from groq import Groq

from .config import settings
from .embedding import cache
from .gemini_client import call_with_gemini_key
from .retry import retry_on_transient

NO_CONTEXT_MESSAGE = "I could not find relevant context in the indexed documents to answer this question."


@dataclass
class GenerationResult:
    answer: str
    token_usage: int = 0
    provider: str = ""
    model: str = ""
    skipped_llm: bool = False


def _cache_key(prompt: str, provider: str, model: str) -> str:
    raw = f"generation:{provider}:{model}:{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _usage_from_gemini(response) -> int:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0
    return int(getattr(usage, "total_token_count", 0) or 0)


def _gemini_generate(prompt: str) -> GenerationResult:
    response = call_with_gemini_key(
        "generation",
        lambda client: client.models.generate_content(
            model=settings.generation_model,
            contents=prompt,
        ),
    )
    return GenerationResult(
        answer=response.text or "",
        token_usage=_usage_from_gemini(response),
        provider="gemini",
        model=settings.generation_model,
    )


def _groq_generate(prompt: str) -> GenerationResult:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required when GENERATION_PROVIDER=groq.")
    client = Groq(api_key=settings.groq_api_key)
    response = retry_on_transient(
        lambda: client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "You answer only from provided context and cite chunk IDs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        ),
        purpose="generation",
        max_retries=max(1, settings.gemini_max_retries + 1),
    )
    usage = getattr(response, "usage", None)
    return GenerationResult(
        answer=response.choices[0].message.content or "",
        token_usage=int(getattr(usage, "total_tokens", 0) or 0),
        provider="groq",
        model=settings.groq_model,
    )


def has_relevant_context(retrieved_chunks: list[dict]) -> bool:
    if not retrieved_chunks:
        return False
    if settings.max_retrieval_distance is None:
        return True

    # LanceDB returns "_distance"; lower distance means more similar.
    distance = retrieved_chunks[0].get("_distance")
    if distance is None:
        return True
    return float(distance) <= settings.max_retrieval_distance


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_text = ""
    for chunk in retrieved_chunks:
        context_text += (
            f'<chunk id="{chunk["id"]}" source="{chunk.get("source_file", "")}" '
            f'index="{chunk.get("chunk_index", "")}">\n{chunk["text"]}\n</chunk>\n\n'
        )

    return f"""You are a careful retrieval-augmented QA assistant.
Answer the question using only the context chunks below.
If the context does not contain the answer, respond exactly: "{NO_CONTEXT_MESSAGE}"
Cite every factual claim with chunk IDs in square brackets, for example [abc123].
Do not cite chunks that do not support the claim.

Context:
{context_text}

Question:
{query}

Answer:"""


def generate_answer(query: str, retrieved_chunks: list[dict], use_cache: bool = True) -> GenerationResult:
    """
    Generates a grounded answer given the query and retrieved context chunks.
    Handles 'no relevant context' appropriately.

    Set ``use_cache=False`` to bypass the cache read so latency can be measured
    cold (the result is still cached for later warm calls).
    """
    provider = settings.generation_provider.lower()
    model = settings.groq_model if provider == "groq" else settings.generation_model

    if not has_relevant_context(retrieved_chunks):
        return GenerationResult(
            answer=NO_CONTEXT_MESSAGE,
            provider=provider,
            model=model,
            skipped_llm=True,
        )

    prompt = build_prompt(query, retrieved_chunks)
    key = _cache_key(prompt, provider, model)
    if use_cache and key in cache:
        return GenerationResult(**cache[key])

    if provider == "mock":
        result = GenerationResult(
            answer=f"Mock answer generated from {len(retrieved_chunks)} retrieved chunks [{retrieved_chunks[0]['id']}].",
            provider="mock",
            model="mock",
        )
    elif provider == "gemini":
        result = _gemini_generate(prompt)
    elif provider == "groq":
        result = _groq_generate(prompt)
    else:
        raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")

    cache[key] = result.__dict__
    return result
