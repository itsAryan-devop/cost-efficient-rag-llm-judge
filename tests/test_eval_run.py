from eval.run import _judge_context


def test_judge_context_uses_cited_chunks_first():
    cited_id = "a" * 64
    other_id = "b" * 64
    answer = f"The answer is supported [{cited_id}]."
    sources = [
        {"id": cited_id, "source_file": "one.md", "text": "cited text"},
        {"id": other_id, "source_file": "two.md", "text": "other text"},
    ]

    context = _judge_context(answer, sources)

    assert "cited text" in context
    assert "other text" not in context


def test_judge_context_falls_back_to_all_sources_without_citations():
    sources = [
        {"id": "a" * 64, "source_file": "one.md", "text": "first text"},
        {"id": "b" * 64, "source_file": "two.md", "text": "second text"},
    ]

    context = _judge_context("No citation here.", sources)

    assert "first text" in context
    assert "second text" in context
