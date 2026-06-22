"""Shared pytest fixtures.

The query-path tests must not depend on a developer-local ``db/`` directory.
``mock_corpus_db`` ingests a tiny fixture corpus into a temp LanceDB using mock
embeddings, and repoints the disk caches at temp dirs, so the suite is fully
hermetic: ``rm -rf db cache && pytest`` is green on a clean checkout.
"""

from __future__ import annotations

import diskcache
import pytest

import src.embedding as embedding
import src.generation as generation
from src.config import settings
from src.ingest import run_ingest

FIXTURE_DOC = """# Fixture Notes

Chunk size and chunk overlap are configurable. The default chunk overlap is 200 characters.

A vector store keeps embeddings so similar text can be retrieved by nearest neighbor search.

The service exposes health, ingest, and query endpoints.
"""


@pytest.fixture
def mock_corpus_db(tmp_path, monkeypatch):
    """Ingest a tiny corpus into an isolated temp DB/cache with mock embeddings."""
    monkeypatch.setattr(settings, "embedding_provider", "mock")
    monkeypatch.setattr(settings, "generation_provider", "mock")
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "lancedb"))

    # Repoint the module-level disk caches so tests never touch the repo cache/.
    temp_cache = diskcache.Cache(str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cache_path", str(tmp_path / "cache"))
    monkeypatch.setattr(embedding, "cache", temp_cache)
    monkeypatch.setattr(generation, "cache", temp_cache)

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "fixture.md").write_text(FIXTURE_DOC, encoding="utf-8")
    monkeypatch.setattr(settings, "data_root", str(data_dir))

    result = run_ingest(str(data_dir))
    assert result["chunks_processed"] > 0
    yield
    temp_cache.close()
