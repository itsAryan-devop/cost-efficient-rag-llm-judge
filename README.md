# Cost-Efficient RAG Application

Problem 1 implementation for the Applied AI / ML Engineering take-home assignment.

This project is a local, low-cost RAG service over PDF, HTML, Markdown, and text files. It uses FastAPI for the HTTP API, LanceDB for embedded vector search, Gemini embeddings by default, and optional Gemini/Groq generation.

## Why This Design

The assignment asks for a credible low-cost alternative to an always-on managed vector database. LanceDB fits that goal because it runs embedded in the app, stores vectors and metadata together, supports metadata filtering, and avoids a separate database server for small to mid-scale RAG workloads.

Accepted trade-offs:

- Embedded LanceDB is excellent for local/single-service deployments but not the best choice for many concurrent writers.
- Managed services still win when the system needs multi-region availability, automatic backups, team operations, or high-concurrency scaling.
- This implementation prefers explicit code over heavy RAG frameworks so ingestion, retrieval, cost, and evaluation stay auditable.

## Project Layout

```text
src/
  api.py             FastAPI endpoints
  config.py          Environment-based config
  ingestion.py       PDF/HTML/MD/TXT parsing, normalization, chunking, hashes
  embedding.py       Gemini/mock embeddings with disk cache
  storage.py         LanceDB table, idempotent upsert, metadata filters
  generation.py      Grounded answer generation with citations
  logger.py          Structured JSON telemetry
eval/
  ir_metrics.py      Recall@k, hit rate, MRR, nDCG@k, context precision
  llm_judge.py       Faithfulness and relevance judge
  run.py             Evaluation runner
  export_chunks.py   Chunk inventory for labeling evaluation questions
  cost_analysis.py   Reproducible cost comparison
data/                Corpus files
reports/             Generated evaluation/cost/chunk reports
tests/               Pytest tests for critical behavior
```

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your keys. Do not commit `.env`.

For Gemini, the app supports key rotation:

```env
GEMINI_API_KEY=primary_key_here
GEMINI_API_KEYS=primary_key_here,backup_key_here
```

If one Gemini key hits a quota/rate-limit error, the app tries the next key and logs only the key index, never the key value.

Recommended free/low-cost config:

```env
DATA_ROOT=data
EMBEDDING_PROVIDER=gemini
GENERATION_PROVIDER=groq
JUDGE_PROVIDER=groq
EMBEDDING_MODEL=gemini-embedding-2
GENERATION_MODEL=gemini-2.5-flash
GROQ_MODEL=llama-3.3-70b-versatile
```

Gemini is used for embeddings. Groq is used for answer generation and judging in the current run because the Gemini free-tier generation quota was exhausted during evaluation. The providers are independently configurable.

For free local smoke tests without API calls:

```env
EMBEDDING_PROVIDER=mock
GENERATION_PROVIDER=mock
JUDGE_PROVIDER=mock
DB_PATH=db/mock_lancedb
CACHE_PATH=cache/mock_diskcache
```

## Run The API

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

- `GET /health`
- `POST /ingest` with `{"data_dir": "data"}`
- `POST /query` with `{"query": "...", "top_k": 5}`
- Metadata filter example: `{"query": "...", "metadata_filter": {"doc_type": "pdf"}}`

No-context handling is controlled by `MIN_RELEVANCE_SCORE`. LanceDB distance is lower-is-better, and the default `0.90` skips the LLM when the nearest chunk is too far away. This prevents unsupported answers and saves tokens.

The API validates empty queries, invalid `top_k` values, and unsupported metadata filters so bad requests return clear client errors.

API-triggered ingestion is restricted to `DATA_ROOT` so a user cannot accidentally point the service at a huge or sensitive folder outside the project.

## Evaluation Workflow

The included corpus in `data/` intentionally covers all required ingestion formats:

- `assignment.pdf` for PDF ingestion
- `cost_analysis_notes.html` for HTML ingestion
- `rag_architecture_notes.md` and `sample_rag_notes.md` for Markdown ingestion

The included `eval/test_set.json` contains 15 fixed questions labeled with relevant chunk IDs from this corpus.

1. Put final corpus files into `data/`.
2. Run ingestion.
3. Export chunk IDs:

```powershell
.\.venv\Scripts\python.exe -m eval.export_chunks
```

4. Use `reports/chunk_inventory.json` to label 15-30 evaluation questions in `eval/test_set.json`.
5. Run evaluation:

```powershell
.\.venv\Scripts\python.exe -m eval.run
```

The evaluation report is saved to:

```text
reports/evaluation_results.json
```

Latest evaluation summary:

| Metric | Value |
|---|---:|
| Cases | 15 |
| Recall@5 | 0.967 |
| Hit Rate | 1.000 |
| MRR | 0.913 |
| nDCG@5 | 0.916 |
| Context Precision | 0.293 |
| Faithfulness | 1.000 |
| Answer Relevance | 1.000 |
| p50 Total Latency (cached eval rerun) | 16 ms |
| p95 Total Latency (cached eval rerun) | 30 ms |
| p50 Retrieval Latency | 16 ms |
| p95 Retrieval Latency | 29 ms |
| Total Generation Tokens | 21088 |

The latest cached rerun records embedding, retrieval, and generation latency separately in `reports/evaluation_results.json`; retrieval latency is the number used for vector-store speed discussion. Token usage is retained from the original generated answers cached during evaluation.

Metrics included:

- Retrieval: Recall@5, hit rate, MRR, nDCG@5, context precision
- Answer quality: faithfulness/groundedness and answer relevance
- Operations: latency and token usage

## Cost Analysis

Generate the cost report:

```powershell
.\.venv\Scripts\python.exe -m eval.cost_analysis
```

Output:

```text
reports/cost_analysis.json
```

Current assumptions:

- 768-dimensional embeddings
- 4 bytes per vector float
- 700 bytes metadata/index overhead per vector
- Local storage: `$0.08/GB/month`
- Managed vector DB baseline estimate: `$70/month`
- One-time embedding estimate: `$0.15 / 1M input tokens`
- Average chunk size estimate: 180 tokens

| Vectors | Est. Index Size (GB) | LanceDB Storage ($/mo) | Managed DB Est. ($/mo) | One-time Embedding Cost ($) |
|---:|---:|---:|---:|---:|
| 100,000 | 0.351 | 0.03 | 70.0 | 2.7 |
| 1,000,000 | 3.513 | 0.28 | 70.0 | 27.0 |
| 10,000,000 | 35.129 | 2.81 | 160.0 | 270.0 |

## Telemetry

Each query logs one JSON line:

```json
{
  "event": "query",
  "query": "What is the default chunk overlap?",
  "latency_ms": 70.16,
  "chunk_count": 1,
  "token_usage": 0,
  "provider": "mock",
  "model": "mock",
  "skipped_llm": false,
  "embedding_latency_ms": 10.2,
  "retrieval_latency_ms": 4.1,
  "generation_latency_ms": 55.8
}
```

## Rubric Evidence

Correctness and ingestion:

- `src/ingestion.py`
- `src/storage.py`
- idempotent test: ingest twice, row count remains stable
- `tests/test_ingestion.py`

Retrieval evaluation:

- `eval/ir_metrics.py`
- `eval/run.py`
- `reports/evaluation_results.json`

Answer evaluation:

- `eval/llm_judge.py`
- faithfulness and relevance fields in `reports/evaluation_results.json`

Cost analysis:

- `eval/cost_analysis.py`
- `reports/cost_analysis.json`
- cost table above

Engineering and clarity:

- environment-only config
- `.env.example`
- `.gitignore`
- JSON logging
- tests
- this README

## Discussion

Retrieval was the stronger layer in the current run. The system found at least one relevant chunk for every evaluation question, with high MRR and nDCG. Context precision is lower because `top_k=5` returns several neighboring chunks from a small corpus. In a larger production corpus, this would be tuned by lowering `top_k`, adding a distance threshold, reranking, or tightening chunk sizes.

Generation was reliable in this small evaluation because the prompt required citations and the judge found the answers faithful and relevant. The main operational weakness was provider quota: Gemini free-tier generation hit its request limit, so the run switched to Groq for generation and judging. This is why the project keeps embedding, generation, and judging providers independently configurable.

I would switch back to a managed vector database when the workload needs many concurrent writers, multi-region availability, automatic backups, strict uptime guarantees, or team operations around monitoring and scaling. For a lightly queried, cost-sensitive RAG index, embedded LanceDB is a credible lower-cost choice.

The accepted trade-off is operational ownership. LanceDB reduces infrastructure cost, but the application owner must manage storage, backups, deployment, and scaling behavior.

## Local Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m eval.cost_analysis
```

Current test status:

```text
18 passed
```
