from src.ingestion import compute_chunk_id, compute_document_hash, compute_document_id, normalize_text


def test_normalize_text_removes_empty_lines():
    assert normalize_text(" hello\n\n world \r\n") == "hello\nworld"


def test_chunk_id_is_stable_for_same_inputs():
    doc_id = compute_document_id("guide.md")

    first = compute_chunk_id(doc_id, 0, "same chunk")
    second = compute_chunk_id(doc_id, 0, "same chunk")

    assert first == second


def test_chunk_id_changes_when_chunk_text_changes():
    doc_id = compute_document_id("guide.md")

    first = compute_chunk_id(doc_id, 0, "same chunk")
    second = compute_chunk_id(doc_id, 0, "different chunk")

    assert first != second


def test_document_id_is_stable_by_path_while_hash_tracks_content():
    assert compute_document_id("guide.md") == compute_document_id("guide.md")
    assert compute_document_hash("old") != compute_document_hash("new")


def test_normalize_text_repairs_common_pdf_mojibake():
    dirty = "Problem 1 \u00e2\u0080\u0094 Cost \u00e2\u0086\u0092 Value \u00c2\u00b7 Item"

    assert normalize_text(dirty) == "Problem 1 - Cost -> Value - Item"
