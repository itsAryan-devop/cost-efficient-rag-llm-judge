"""Reusable ingestion pipeline shared by the API, CLI, and tests.

Parsing/chunking lives in :mod:`src.ingestion`; embedding lives in
:mod:`src.embedding`; storage lives in :mod:`src.storage`. This module wires
them together so there is a single ingestion code path.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import settings
from .embedding import get_embedding
from .ingestion import process_documents
from .storage import upsert_vectors


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for chunk in chunks:
        chunk["vector"] = get_embedding(chunk["text"], input_type="document")
        chunk["embedding_model"] = (
            "mock" if settings.embedding_provider.lower() == "mock" else settings.embedding_model
        )
        chunk["embedding_dimension"] = settings.embedding_dimension
    return chunks


def run_ingest(data_dir: str | None = None) -> Dict[str, Any]:
    """Parse, chunk, embed and upsert every supported document under ``data_dir``."""
    data_dir = data_dir or settings.data_root
    chunks = process_documents(data_dir)
    if not chunks:
        return {"status": "success", "message": "No documents found or processed.", "chunks_processed": 0}

    embed_chunks(chunks)
    upsert_vectors(chunks)
    return {"status": "success", "chunks_processed": len(chunks)}


def main() -> None:
    result = run_ingest()
    print(result)


if __name__ == "__main__":
    main()
