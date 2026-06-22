from src.config import settings
from src.storage import build_where, get_table, upsert_vectors


def test_build_where_allows_only_supported_metadata_fields():
    assert build_where({"doc_type": "pdf"}) == "doc_type = 'pdf'"


def test_lancedb_upsert_is_idempotent(tmp_path):
    original_db_path = settings.db_path
    settings.db_path = str(tmp_path / "lancedb")
    try:
        row = {
            "id": "chunk-1",
            "document_id": "doc-1",
            "text": "hello world",
            "vector": [0.0] * settings.embedding_dimension,
            "metadata": {
                "source_file": "sample.md",
                "doc_type": "md",
                "chunk_index": 0,
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        upsert_vectors([row])
        upsert_vectors([row])

        assert get_table().count_rows() == 1
    finally:
        settings.db_path = original_db_path
