from eval.ir_metrics import (
    average_precision,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ir_metrics_with_relevant_chunk_at_rank_two():
    retrieved = ["a", "b", "c"]
    relevant = ["b"]

    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate(retrieved, relevant) == 1
    assert hit_rate(retrieved, relevant, k=1) == 0
    assert hit_rate(retrieved, relevant, k=2) == 1
    assert mrr(retrieved, relevant) == 0.5
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert 0 < ndcg_at_k(retrieved, relevant, k=3) < 1


def test_ir_metrics_with_no_hit():
    retrieved = ["a", "b", "c"]
    relevant = ["x"]

    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert hit_rate(retrieved, relevant) == 0
    assert mrr(retrieved, relevant) == 0.0
    assert precision_at_k(retrieved, relevant, k=3) == 0.0
    assert ndcg_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k_respects_k_window():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["a", "c"]

    # Only the first two are considered; one of them is relevant -> 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    # Over all four, two of four are relevant -> 0.5
    assert precision_at_k(retrieved, relevant) == 0.5


def test_average_precision_rewards_higher_ranking():
    relevant = ["a", "b"]
    # Both relevant items ranked first: AP = (1/1 + 2/2) / 2 = 1.0
    assert average_precision(["a", "b", "c"], relevant, k=3) == 1.0
    # Same items ranked lower scores less than the ideal ranking.
    worse = average_precision(["c", "a", "b"], relevant, k=3)
    assert worse < 1.0
    # No relevant retrieved -> 0.0
    assert average_precision(["x", "y"], relevant, k=2) == 0.0
