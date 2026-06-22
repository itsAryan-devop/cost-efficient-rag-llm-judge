from src.ingestion import compute_chunk_id, compute_document_id, normalize_text


def test_normalize_text_removes_empty_lines():
    assert normalize_text(" hello\n\n world \r\n") == "hello\nworld"


def test_chunk_id_is_stable_for_same_inputs():
    doc_id = compute_document_id("guide.md", "same text")

    first = compute_chunk_id(doc_id, 0, "same chunk")
    second = compute_chunk_id(doc_id, 0, "same chunk")

    assert first == second


def test_chunk_id_changes_when_chunk_text_changes():
    doc_id = compute_document_id("guide.md", "same text")

    first = compute_chunk_id(doc_id, 0, "same chunk")
    second = compute_chunk_id(doc_id, 0, "different chunk")

    assert first != second
