import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .embedding import get_embedding, is_cached
from .generation import generate_answer
from .ingest import embed_chunks
from .ingestion import process_documents
from .logger import log_query
from .storage import get_table, search, upsert_vectors

app = FastAPI(title="Cost-Efficient RAG Application")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=50)
    metadata_filter: dict[str, str] | None = None


class IngestRequest(BaseModel):
    data_dir: str = settings.data_root


class SourceChunk(BaseModel):
    id: str
    document_id: str
    document_hash: str
    embedding_model: str
    embedding_dimension: int
    text: str
    source_file: str
    doc_type: str
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    distance: float | None = Field(default=None, alias="_distance")

    model_config = {"populate_by_name": True}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    token_usage: int
    provider: str
    model: str


def resolve_ingest_dir(data_dir: str) -> str:
    """Restrict API-triggered ingestion to DATA_ROOT to avoid accidental broad scans."""
    root = Path(settings.data_root).resolve()
    requested = Path(data_dir).resolve()

    if requested != root and root not in requested.parents:
        raise ValueError(f"data_dir must be inside configured DATA_ROOT: {settings.data_root}")
    if not requested.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")
    if not requested.is_dir():
        raise ValueError(f"data_dir must be a directory: {data_dir}")

    return str(requested)


@app.get("/health")
def health_check():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    """Readiness: the vector store is reachable; reports row count and embedding info."""
    try:
        table = get_table()
        row_count = table.count_rows()
        embedding_model = settings.embedding_model
        embedding_dimension = settings.embedding_dimension
        if row_count > 0:
            sample = table.to_arrow().to_pylist()[0]
            embedding_model = sample.get("embedding_model", embedding_model)
            embedding_dimension = sample.get("embedding_dimension", embedding_dimension)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc

    return {
        "status": "ready" if row_count > 0 else "empty",
        "table": "documents",
        "row_count": row_count,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_provider": settings.embedding_provider,
    }


@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    """Idempotent document ingestion."""
    try:
        data_dir = resolve_ingest_dir(req.data_dir)
        chunks = process_documents(data_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        return {"status": "success", "message": "No documents found or processed.", "chunks_processed": 0}

    try:
        embed_chunks(chunks)
        upsert_vectors(chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

    return {"status": "success", "chunks_processed": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    start_time = time.time()

    cached = is_cached(req.query, input_type="query")
    try:
        # 1. Embed query
        embedding_start = time.time()
        query_vector = get_embedding(req.query, input_type="query")
        embedding_latency_ms = (time.time() - embedding_start) * 1000

        # 2. Retrieve
        retrieval_start = time.time()
        results = search(query_vector, top_k=req.top_k, metadata_filter=req.metadata_filter)
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000

        # 3. Generate
        generation_start = time.time()
        generation = generate_answer(req.query, results)
        generation_latency_ms = (time.time() - generation_start) * 1000
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc

    latency = (time.time() - start_time) * 1000

    log_query(
        req.query,
        latency,
        len(results),
        generation.token_usage,
        provider=generation.provider,
        model=generation.model,
        skipped_llm=generation.skipped_llm,
        cached=cached,
        embedding_latency_ms=round(embedding_latency_ms, 2),
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
    )

    return {
        "answer": generation.answer,
        "sources": results,
        "latency_ms": latency,
        "embedding_latency_ms": embedding_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "token_usage": generation.token_usage,
        "provider": generation.provider,
        "model": generation.model,
    }
