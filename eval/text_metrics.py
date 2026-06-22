"""SQuAD-style Exact Match (EM) and token-level F1 against reference answers.

Normalization follows the SQuAD convention: lowercase, strip punctuation, drop
articles (a/an/the), and collapse whitespace. These metrics let us score answers
against gold ``reference_answer`` text without an LLM, which keeps the evaluation
cheap and deterministic.
"""

from __future__ import annotations

import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, drop articles, and collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized prediction equals the normalized reference, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)

    # Two empty strings match exactly; one empty means no overlap.
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
