"""Tests for the shared retry + key-rotation helper used by the Groq paths."""
from __future__ import annotations

import pytest

from src.retry import call_with_key_rotation, split_keys


class _FakeRateLimit(Exception):
    status_code = 429


class _FakeAuthError(Exception):
    status_code = 401


class _FakeNonRetryable(Exception):
    status_code = 400


def test_split_keys_handles_commas_semicolons_and_newlines():
    assert split_keys("a, b ,, c\nd;e ") == ["a", "b", "c", "d", "e"]
    assert split_keys("") == []
    assert split_keys(None) == []


def test_call_with_key_rotation_rotates_on_rate_limit():
    attempts: list[str] = []

    def make_client(key: str) -> str:
        return key  # client is the key string in this fake

    def op(client: str) -> str:
        attempts.append(client)
        if client == "k1":
            raise _FakeRateLimit("429")
        return f"ok:{client}"

    out = call_with_key_rotation(["k1", "k2"], make_client, op, purpose="t", max_retries=1)
    assert out == "ok:k2"
    assert attempts == ["k1", "k2"]


def test_call_with_key_rotation_rotates_on_auth_error():
    """401/403 indicate a bad key, not a transient fault, so we rotate."""
    attempts: list[str] = []

    def make_client(key: str) -> str:
        return key

    def op(client: str) -> str:
        attempts.append(client)
        if client == "expired":
            raise _FakeAuthError("401")
        return "ok"

    out = call_with_key_rotation(["expired", "good"], make_client, op, purpose="t", max_retries=1)
    assert out == "ok"
    assert attempts == ["expired", "good"]


def test_call_with_key_rotation_does_not_rotate_on_4xx_other_than_auth_rate():
    """A 400 is a real bug — don't burn keys retrying it."""
    attempts: list[str] = []

    def op(client: str) -> str:
        attempts.append(client)
        raise _FakeNonRetryable("400")

    with pytest.raises(_FakeNonRetryable):
        call_with_key_rotation(["k1", "k2"], lambda k: k, op, purpose="t", max_retries=1)
    assert attempts == ["k1"]  # only the first key tried


def test_call_with_key_rotation_empty_keys_raises():
    with pytest.raises(RuntimeError, match="no API keys"):
        call_with_key_rotation([], lambda k: k, lambda c: "ok", purpose="t")


def test_call_with_key_rotation_all_keys_exhausted_raises_last_error():
    def op(client: str) -> str:
        raise _FakeRateLimit("429 on every key")

    with pytest.raises(_FakeRateLimit):
        call_with_key_rotation(["k1", "k2"], lambda k: k, op, purpose="t", max_retries=1)
