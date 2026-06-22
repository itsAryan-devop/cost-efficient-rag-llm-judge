import re
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai.errors import ClientError, ServerError

from .config import settings
from .logger import log_event


T = TypeVar("T")
ROTATABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}


def get_gemini_api_keys() -> list[str]:
    """Returns unique Gemini keys from GEMINI_API_KEYS plus GEMINI_API_KEY."""
    candidates: list[str] = []

    if settings.gemini_api_keys:
        candidates.extend(
            key.strip()
            for key in re.split(r"[,;\n]+", settings.gemini_api_keys)
            if key.strip()
        )

    if settings.gemini_api_key:
        candidates.append(settings.gemini_api_key.strip())

    unique_keys: list[str] = []
    seen = set()
    for key in candidates:
        if key and key not in seen:
            unique_keys.append(key)
            seen.add(key)

    return unique_keys


def call_with_gemini_key(purpose: str, operation: Callable[[genai.Client], T]) -> T:
    """
    Runs a Gemini operation with key rotation.

    The key itself is never logged. We only log the zero-based key index and
    status code so runs remain auditable without leaking secrets.
    """
    keys = get_gemini_api_keys()
    if not keys:
        raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEYS for Gemini calls.")

    last_error: Exception | None = None
    for key_index, api_key in enumerate(keys):
        client = genai.Client(api_key=api_key)
        try:
            return operation(client)
        except (ClientError, ServerError) as exc:
            status_code = int(getattr(exc, "status_code", 0) or 0)
            last_error = exc
            should_rotate = status_code in ROTATABLE_STATUS_CODES and key_index < len(keys) - 1
            log_event(
                "gemini_call_failed",
                purpose=purpose,
                key_index=key_index,
                status_code=status_code,
                rotated=should_rotate,
            )
            if should_rotate:
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Gemini operation failed without a captured error.")
