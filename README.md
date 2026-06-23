# Applied AI / ML Engineering Take-Home

This repository contains both assignment problems:

- Problem 1: Cost-Efficient RAG Application
- Problem 2: LLM-as-Judge Evaluation Pipeline

The default production-style provider order is:

- Embeddings: Gemini, with `GEMINI_API_KEYS` rotation.
- Generation: Gemini first, Groq fallback, with both Gemini and Groq key pools.
- Judging: Gemini first, Groq fallback, independently configurable from generation.
- Offline CI/evaluation mode: `mock` providers, so the project remains reproducible without spending API quota.

No API keys are committed. `.env` is gitignored.

## Problem 1: RAG Service

The RAG app is a FastAPI service over a PDF / HTML / Markdown corpus, backed by embedded LanceDB.

Why LanceDB:

- It runs inside the app process and avoids a separate always-on vector DB service.
- It stores vectors and metadata together.
- It supports metadata filtering.
- It is a credible low-cost choice for a lightly queried, single-service RAG index.

Accepted trade-off: managed vector DBs are better once you need multi-region availability, managed backups, high-concurrency writes, strict SLAs, or team operations around scaling.

## Problem 1 Evidence

Offline reproducible evaluation:

- File: `reports/evaluation_results.json`
- Run: `python -m eval.run`
- Providers: mock embedding/generation/judge for quota-free repeatability
- Test set: 26 questions, including 23 answerable and 3 out-of-corpus refusal cases

Latest offline retrieval metrics:

| Metric | Value |
|---|---:|
| Recall@5 | 0.826 |
| Hit Rate | 0.826 |
| MRR | 0.592 |
| nDCG@5 | 0.651 |
| Precision@5 | 0.165 |
| Average Precision | 0.592 |

Answer-quality evidence:

- The answer judge scores faithfulness and relevance on a 1-5 rubric.
- The offline system answers are mock placeholders, so their EM/F1 and per-case faithfulness are intentionally low.
- To prove the judge is not a rubber stamp, the harness includes adversarial probes:
  - Correct grounded answer faithfulness: 5 / 5
  - Confidently wrong answer faithfulness: 1 / 5
- Offline refusal accuracy: 1.0 on the 3 out-of-corpus cases.

Live real-provider smoke:

- File: `reports/smoke_results.json`
- Run: `python -m eval.smoke`
- Providers used in this run: Gemini primary, Groq fallback
- Gemini key rotation was exercised; when Gemini hit 429/503, Groq fallback completed generation/judging.

Smoke results:

| Check | Result |
|---|---:|
| Grounded answer: HTTP 502 | faithfulness 5 / 5, relevance 5 / 5 |
| Grounded answer: FastAPI standards | faithfulness 5 / 5, relevance 5 / 5 |
| Out-of-corpus question | refused correctly |

Latency from the smoke is dominated by external API calls. Retrieval itself remains small:

| Query | Embedding | Retrieval | Generation | Total |
|---|---:|---:|---:|---:|
| HTTP 502 | 1454 ms | 62 ms | 6492 ms | 8009 ms |
| FastAPI standards | 1441 ms | 19 ms | 13427 ms | 14886 ms |
| France refusal | 1328 ms | 12 ms | 11155 ms | 12495 ms |

The France case returned the correct no-context answer. In this smoke, it was refused by the grounded prompt fallback rather than the early distance gate, so the report records `skipped_llm: false` but `refused: true`.

## Problem 1 Requirements Checklist

- PDF/HTML/MD ingestion: implemented in `src/ingestion.py`.
- Configurable chunk size and overlap: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` by default.
- Idempotent re-ingest: deletes by stable `document_id` and chunk ID before insert.
- Embedding metadata: stores model and dimensionality per chunk.
- Vectors plus metadata: LanceDB schema includes vector, text, source file, doc type, hashes, chunk metadata.
- Metadata filter: `/query` supports allowlisted filters such as `doc_type`.
- Top-k retrieval: `top_k` is an API parameter.
- Grounded answers with citations: prompt requires chunk-ID citations.
- No-context handling: early distance gate plus prompt-level fallback.
- HTTP endpoint: FastAPI `/health`, `/ready`, `/ingest`, `/query`.
- Env config and no hardcoded secrets: `pydantic-settings`, `.env.example`, `.gitignore`.
- Logging: per-query latency, chunk count, token usage, provider/model, cache status.
- Retrieval eval: Recall@k, Hit Rate, MRR, nDCG@k, Precision@k, Average Precision.
- Answer eval: EM, token-F1, faithfulness, relevance, adversarial judge probes.
- Cost and latency eval: reports include cost table and p50/p95 retrieval latency.

## Cost Analysis

File: `reports/cost_analysis.json`

Assumptions:

- 768-dimensional embeddings
- 4 bytes per float
- 700 bytes metadata/index overhead per vector
- LanceDB host estimate: $10/month
- Local storage: $0.08/GB/month
- Managed baseline: Pinecone serverless Standard plan, priced at $50/month minimum plus storage
- Embedding estimate: $0.15 / 1M input tokens
- Chunk size is characters; token estimate uses about 4.4 chars/token

| Vectors | Index GB | LanceDB total/month | Managed total/month | One-time embedding |
|---:|---:|---:|---:|---:|
| 100,000 | 0.351 | $10.03 | $50.00 | $2.73 |
| 1,000,000 | 3.513 | $10.28 | $50.00 | $27.27 |
| 10,000,000 | 35.129 | $12.81 | $50.00 | $272.73 |

## Problem 2: LLM-as-Judge Pipeline

Problem 2 lives under `eval/pipeline/`.

It implements:

- YAML/JSON test suite input.
- Pairwise A/B judging.
- Structured Pydantic verdicts with per-criterion scores and rationales.
- Robust JSON extraction and retry on malformed responses.
- Full JSONL audit log containing judge prompt, raw response, parsed verdict, tokens, cost, latency.
- Independent judge/generator config.
- Position-bias mitigation by running both A/B and B/A orderings.
- Verbosity and sycophancy probes.
- Self-enhancement warning when judge and generator families match.
- Test-retest validation.
- Gold agreement and Cohen's kappa when gold labels exist.

Headline committed Problem 2 result:

- File: `reports/p2_evaluation_report.json`
- Audit log: `reports/p2_audit_log.jsonl`
- Suite: `RAG Quality Comparison`
- Judge: Groq `llama-3.3-70b-versatile`
- Winner: `Prompt v1 (concise)`

| Metric | Value |
|---|---:|
| Cases | 5 |
| Win rate A / B | 0.80 / 0.20 |
| Weighted score A / B | 4.850 / 2.910 |
| Pass rate A / B (>=4.0) | 1.00 / 0.40 |
| Position bias rate | 0.00 |
| Position flip rate | 0.00 |
| Verbosity probe / Sycophancy probe | PASSED / PASSED |
| Adversarial probes passed | 3 / 3 |
| Gold agreement | 0.60 |
| Cohen's kappa | 0.231 |
| Test-retest agreement (n=3) | 1.00 |
| Error rate | 0.00 |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
cp .env.example .env
```

Fill `.env` with keys if using real providers. For free offline verification, set:

```env
EMBEDDING_PROVIDER=mock
GENERATION_PROVIDER=mock
JUDGE_PROVIDER=mock
```

Recommended real-provider config:

```env
EMBEDDING_PROVIDER=gemini
GENERATION_PROVIDER=gemini
GENERATION_FALLBACK_PROVIDER=groq
JUDGE_PROVIDER=gemini
JUDGE_FALLBACK_PROVIDER=groq
GEMINI_API_KEYS=key1,key2,key3
GROQ_API_KEYS=key1,key2
```

## Run

```bash
python -m src.ingestion
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

```bash
curl localhost:8000/health
curl localhost:8000/ready
curl -X POST localhost:8000/ingest -H "content-type: application/json" -d "{\"data_dir\":\"data/corpus\"}"
curl -X POST localhost:8000/query -H "content-type: application/json" -d "{\"query\":\"What does HTTP status code 502 mean?\",\"top_k\":5}"
```

Metadata filter example:

```json
{"query": "What does HTTP status code 502 mean?", "metadata_filter": {"doc_type": "pdf"}}
```

## Reproduce Reports

```bash
python -m eval.export_chunks
python -m eval.build_test_set
python -m eval.run
python -m eval.cost_analysis
python -m eval.smoke
python -m eval.run_pipeline --suite eval/suites/sample_suite.yaml
```

`eval.smoke` and real Problem 2 judging consume API quota. The offline tests and mock evaluation do not.

## Engineering Notes

- API-triggered ingestion is restricted to `DATA_ROOT`.
- Unsupported metadata filter keys return `400`.
- API response models exclude vectors from `/query`.
- Gemini and Groq failures log only key indexes, never key values.
- Provider retries use exponential backoff for 429/5xx.
- Generation and judging can fall back from Gemini to Groq on quota/provider failures.
- Evaluation writes partial results after each case to avoid losing progress on quota errors.

## Verification

Current local verification:

```text
73 passed
```

Run:

```bash
python -m pytest -q
```
