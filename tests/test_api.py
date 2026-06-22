from fastapi.testclient import TestClient

from src.api import app
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
