"""Shared transient-error retry helper used by both the Gemini and Groq paths.

Gemini additionally rotates across an API-key pool (see
:mod:`src.gemini_client`); this module provides the common backoff-on-429/5xx
behaviour so every provider degrades consistently instead of turning a single
rate-limit into an unhandled 502.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from .logger import log_event

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ROTATABLE_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}


def status_code(exc: Exception) -> int:
    """Best-effort extraction of an HTTP status code across SDK exception shapes."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else 0


def _backoff_seconds(attempt: int) -> float:
    return min(2**attempt, 4)


def retry_on_transient(
    operation: Callable[[], T],
    *,
    purpose: str,
    max_retries: int = 3,
) -> T:
    """Run ``operation`` with exponential backoff on retryable (429/5xx) errors.

    Non-retryable errors are raised immediately. The final attempt's error is
    re-raised so callers see a clear failure surface.
    """
    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - re-raised below; we only branch on status
            code = status_code(exc)
            last_error = exc
            retryable = code in RETRYABLE_STATUS_CODES and attempt < attempts - 1
            log_event(
                "provider_call_failed",
                purpose=purpose,
                status_code=code,
                attempt=attempt + 1,
                max_retries=attempts,
                retryable=retryable,
            )
            if retryable:
                time.sleep(_backoff_seconds(attempt))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError(f"{purpose} failed without a captured error.")
