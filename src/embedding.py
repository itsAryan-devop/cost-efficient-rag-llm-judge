import hashlib
import math
import re
from typing import List

from google.genai import types
from diskcache import Cache
from .config import settings
from .gemini_client import call_with_gemini_key

cache = Cache(settings.cache_path)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _cache_key(provider: str, model: str, text: str, input_type: str) -> str:
    raw = f"embedding:{provider}:{model}:{settings.embedding_dimension}:{input_type}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _mock_embedding(text: str) -> List[float]:
    """Cheap deterministic embedding for tests and dry runs."""
    vector = [0.0] * settings.embedding_dimension
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = [text.lower()]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % settings.embedding_dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]

def _format_for_gemini_embedding(text: str, input_type: str) -> str:
    if settings.embedding_model == "gemini-embedding-2":
        if input_type == "query":
            return f"task: question answering | query: {text}"
        return f"title: none | text: {text}"
    return text

def _gemini_embedding(text: str, input_type: str) -> List[float]:
    if not settings.gemini_api_key:
        # GEMINI_API_KEYS may still be present; the rotation helper validates.
        pass
    config = types.EmbedContentConfig(output_dimensionality=settings.embedding_dimension)
    if settings.embedding_model == "gemini-embedding-001":
        config.task_type = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"

    response = call_with_gemini_key(
        "embedding",
        lambda client: client.models.embed_content(
            model=settings.embedding_model,
            contents=_format_for_gemini_embedding(text, input_type),
            config=config,
        ),
    )
    embedding = response.embeddings[0].values
    return [float(v) for v in embedding]

def get_embedding(text: str, input_type: str = "document") -> List[float]:
    """
    Gets the embedding for a piece of text.
    Checks the local disk cache first to save API calls.
    """
    provider = settings.embedding_provider.lower()
    model = "mock" if provider == "mock" else settings.embedding_model
    key = _cache_key(provider, model, text, input_type)
    
    if key in cache:
        return cache[key]
    
    if provider == "mock":
        embedding = _mock_embedding(text)
    elif provider == "gemini":
        embedding = _gemini_embedding(text, input_type)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    if len(embedding) != settings.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.embedding_dimension}, got {len(embedding)}"
        )

    cache[key] = embedding
    return embedding
