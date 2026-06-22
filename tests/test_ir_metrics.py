from eval.ir_metrics import recall_at_k, hit_rate, mrr, context_precision, ndcg_at_k


def test_ir_metrics_with_relevant_chunk_at_rank_two():
    retrieved = ["a", "b", "c"]
    relevant = ["b"]

    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate(retrieved, relevant) == 1
    assert hit_rate(retrieved, relevant, k=1) == 0
    assert hit_rate(retrieved, relevant, k=2) == 1
    assert mrr(retrieved, relevant) == 0.5
    assert context_precision(retrieved, relevant) == 1 / 3
    assert 0 < ndcg_at_k(retrieved, relevant, k=3) < 1


def test_ir_metrics_with_no_hit():
    retrieved = ["a", "b", "c"]
    relevant = ["x"]

    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert hit_rate(retrieved, relevant) == 0
    assert mrr(retrieved, relevant) == 0.0
    assert context_precision(retrieved, relevant) == 0.0
    assert ndcg_at_k(retrieved, relevant, k=3) == 0.0
