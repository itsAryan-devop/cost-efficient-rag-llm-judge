import re
import time
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai.errors import ClientError, ServerError

from .config import settings
from .logger import log_event
from .retry import ROTATABLE_STATUS_CODES, RETRYABLE_STATUS_CODES, status_code as _shared_status_code


T = TypeVar("T")


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


def _status_code(exc: Exception) -> int:
    """Extracts a provider status code across google-genai SDK versions."""
    return _shared_status_code(exc)


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
    max_retries = max(1, settings.gemini_max_retries)
    for key_index, api_key in enumerate(keys):
        client = genai.Client(api_key=api_key)
        for attempt in range(max_retries):
            try:
                return operation(client)
            except (ClientError, ServerError) as exc:
                status_code = _status_code(exc)
                last_error = exc
                retryable = status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1
                should_rotate = (
                    status_code in ROTATABLE_STATUS_CODES
                    and not retryable
                    and key_index < len(keys) - 1
                )
                log_event(
                    "gemini_call_failed",
                    purpose=purpose,
                    key_index=key_index,
                    status_code=status_code,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    rotated=should_rotate,
                )
                if retryable:
                    time.sleep(min(2**attempt, 4))
                    continue
                if should_rotate:
                    break
                raise

    if last_error:
        raise last_error
    raise RuntimeError("Gemini operation failed without a captured error.")
