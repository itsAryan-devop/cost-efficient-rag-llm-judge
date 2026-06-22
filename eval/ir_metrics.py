def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Fraction of relevant chunks retrieved in the top-k results."""
    if not relevant_ids:
        return 0.0
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    relevant_retrieved = sum(1 for r_id in relevant_ids if r_id in candidates)
    return relevant_retrieved / len(relevant_ids)


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> int:
    """Returns 1 if at least one relevant ID is in the retrieved IDs, else 0."""
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    return int(any(r_id in candidates for r_id in relevant_ids))


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, r_id in enumerate(retrieved_ids):
        if r_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Precision@k: fraction of the top-k retrieved chunks that are relevant.

    This is an order-insensitive set metric. (It was previously mislabelled
    ``context_precision``; the real order-aware metric is ``average_precision``.)
    """
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    if not candidates:
        return 0.0
    relevant_retrieved = sum(1 for r_id in candidates if r_id in relevant_ids)
    return relevant_retrieved / len(candidates)


def average_precision(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Order-aware average precision: mean of precision@i over the ranks where a
    relevant chunk appears, normalized by the number of relevant items. This is
    the per-query term of Mean Average Precision (MAP) and rewards ranking
    relevant chunks higher.
    """
    if not relevant_ids:
        return 0.0
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    hits = 0
    score = 0.0
    for i, r_id in enumerate(candidates):
        if r_id in relevant_ids:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant_ids), len(candidates)) if hits else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    import math

    dcg = 0.0
    for i in range(min(k, len(retrieved_ids))):
        if retrieved_ids[i] in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    idcg = 0.0
    for i in range(min(k, len(relevant_ids))):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
