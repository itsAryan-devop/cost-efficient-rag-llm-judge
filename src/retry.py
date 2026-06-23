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


def split_keys(raw: str) -> list[str]:
    """Parse a comma/semicolon/newline-separated key pool, preserving order, dropping blanks."""
    return [k.strip() for k in re.split(r"[,;\n]+", raw or "") if k.strip()]


def call_with_key_rotation(
    keys: list[str],
    make_client: Callable[[str], object],
    operation: Callable[[object], T],
    *,
    purpose: str,
    max_retries: int = 3,
) -> T:
    """Try ``operation(make_client(key))`` across each key; rotate on auth / rate / 5xx
    errors (after this key's own transient retries are exhausted).

    Used by both ``src/generation.py`` and ``eval/pipeline/judge.py`` so the Groq path
    has parity with Gemini's key rotation. Only the zero-based key index is logged,
    never the key itself.
    """
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError(f"{purpose}: no API keys configured")
    last_error: Exception | None = None
    for key_index, key in enumerate(keys):
        client = make_client(key)
        try:
            return retry_on_transient(
                lambda c=client: operation(c),
                purpose=f"{purpose}[key={key_index}]",
                max_retries=max_retries,
            )
        except Exception as exc:
            last_error = exc
            code = status_code(exc)
            should_rotate = code in ROTATABLE_STATUS_CODES and key_index < len(keys) - 1
            log_event(
                "key_rotation",
                purpose=purpose,
                key_index=key_index,
                status_code=code,
                rotated=should_rotate,
            )
            if not should_rotate:
                raise
    raise last_error if last_error else RuntimeError(f"{purpose}: rotation exhausted")
