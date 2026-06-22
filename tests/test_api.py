from fastapi.testclient import TestClient

from src.api import app, resolve_ingest_dir
from src.config import settings


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rejects_empty_query():
    client = TestClient(app)

    response = client.post("/query", json={"query": ""})

    assert response.status_code == 422


def test_query_rejects_invalid_top_k():
    client = TestClient(app)

    response = client.post("/query", json={"query": "hello", "top_k": 0})

    assert response.status_code == 422


def test_query_returns_bad_request_for_unsupported_filter(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(settings, "embedding_provider", "mock")
    response = client.post(
        "/query",
        json={"query": "hello", "metadata_filter": {"unknown": "value"}},
    )

    assert response.status_code == 400
    assert "Unsupported metadata filter" in response.json()["detail"]


def test_query_response_model_excludes_vectors(mock_corpus_db):
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is the default chunk overlap?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"]
    assert "vector" not in body["sources"][0]
    assert "_distance" in body["sources"][0]


def test_query_returns_grounded_sources_after_ingest(mock_corpus_db):
    """End-to-end hermetic check: ingest a temp corpus, then query it."""
    client = TestClient(app)

    response = client.post("/query", json={"query": "nearest neighbor search vector store", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"]
    assert all("vector" not in source for source in body["sources"])
    assert body["provider"] == "mock"


def test_resolve_ingest_dir_rejects_paths_outside_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside"
    data_root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    try:
        resolve_ingest_dir(str(outside))
    except ValueError as exc:
        assert "DATA_ROOT" in str(exc)
    else:
        raise AssertionError("Expected resolve_ingest_dir to reject outside path")


def test_resolve_ingest_dir_accepts_data_root_subdirectory(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    nested = data_root / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setattr(settings, "data_root", str(data_root))

    assert resolve_ingest_dir(str(nested)) == str(nested.resolve())


def test_ingest_rejects_missing_data_dir(tmp_path, monkeypatch):
    client = TestClient(app)
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(settings, "data_root", str(data_root))

    response = client.post("/ingest", json={"data_dir": str(data_root / "missing")})

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
