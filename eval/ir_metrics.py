from typing import List

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int | None = None) -> float:
    """Fraction of relevant chunks retrieved in the top-k results."""
    if not relevant_ids:
        return 0.0
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    relevant_retrieved = sum(1 for r_id in relevant_ids if r_id in candidates)
    return relevant_retrieved / len(relevant_ids)

def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> int:
    """Returns 1 if at least one relevant ID is in the retrieved IDs, else 0."""
    return int(any(r_id in retrieved_ids for r_id in relevant_ids))

def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, r_id in enumerate(retrieved_ids):
        if r_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def context_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Proportion of retrieved chunks that are relevant."""
    if not retrieved_ids:
        return 0.0
    relevant_retrieved = sum(1 for r_id in retrieved_ids if r_id in relevant_ids)
    return relevant_retrieved / len(retrieved_ids)

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
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
