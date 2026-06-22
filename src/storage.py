import os
from collections.abc import Mapping
from typing import Any

import lancedb
import pyarrow as pa

from .config import settings

FILTERABLE_FIELDS = {"source_file", "doc_type", "document_id"}


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("document_hash", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_dimension", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), settings.embedding_dimension)),
            pa.field("text", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("chunk_size", pa.int64()),
            pa.field("chunk_overlap", pa.int64()),
        ]
    )


def get_table():
    """Gets or creates the LanceDB table."""
    os.makedirs(
        os.path.dirname(settings.db_path) if os.path.dirname(settings.db_path) else ".", exist_ok=True
    )
    db = lancedb.connect(settings.db_path)

    return db.create_table("documents", schema=schema(), exist_ok=True)


def flatten_chunk(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "document_hash": row["document_hash"],
        "embedding_model": row.get("embedding_model", settings.embedding_model),
        "embedding_dimension": int(row.get("embedding_dimension", settings.embedding_dimension)),
        "vector": [float(v) for v in row["vector"]],
        "text": row["text"],
        "source_file": metadata.get("source_file", ""),
        "doc_type": metadata.get("doc_type", ""),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "chunk_size": int(metadata.get("chunk_size", settings.chunk_size)),
        "chunk_overlap": int(metadata.get("chunk_overlap", settings.chunk_overlap)),
    }


def upsert_vectors(data: list[dict[str, Any]]):
    """
    Upserts vectors into LanceDB.
    data format:
    [
        {"id": "hash", "vector": [0.1, ...], "text": "chunk text", "metadata": '{"source_file": "..."}'}
    ]
    """
    if not data:
        return

    table = get_table()
    rows = [flatten_chunk(row) for row in data]

    # Delete by stable document_id first so changed documents do not leave stale chunks.
    document_ids = sorted({row["document_id"] for row in rows})
    batch_size = 100
    for i in range(0, len(document_ids), batch_size):
        batch_ids = document_ids[i : i + batch_size]
        ids_str = ", ".join(f"'{x}'" for x in batch_ids)
        table.delete(f"document_id IN ({ids_str})")

    # Delete by chunk ID as a second guard for older rows created by previous schemas.
    ids_to_upsert = [row["id"] for row in rows]
    for i in range(0, len(ids_to_upsert), batch_size):
        batch_ids = ids_to_upsert[i : i + batch_size]
        ids_str = ", ".join(f"'{x}'" for x in batch_ids)
        table.delete(f"id IN ({ids_str})")

    table.add(rows)


def build_where(metadata_filter: Mapping[str, str] | None) -> str | None:
    if not metadata_filter:
        return None

    clauses = []
    for key, value in metadata_filter.items():
        if key not in FILTERABLE_FIELDS:
            raise ValueError(f"Unsupported metadata filter: {key}. Allowed: {sorted(FILTERABLE_FIELDS)}")
        safe_value = str(value).replace("'", "''")
        clauses.append(f"{key} = '{safe_value}'")

    return " AND ".join(clauses) if clauses else None


def search(
    query_vector: list[float], top_k: int = 5, metadata_filter: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    """
    Searches LanceDB for the most similar vectors.
    """
    where_clause = build_where(metadata_filter)
    table = get_table()
    if table.count_rows() == 0:
        return []

    query = table.search(query_vector).limit(top_k)
    if where_clause:
        query = query.where(where_clause)

    results = query.to_list()
    return results
