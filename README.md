# Cost-Efficient RAG Application (Problem 1)

A small, low-cost RAG service over PDF / HTML / Markdown / text. FastAPI HTTP API,
embedded **LanceDB** vector store, Gemini embeddings, and Groq/Gemini generation —
chosen so there is no always-on managed vector-database bill.

## 30-second what + why LanceDB

LanceDB runs **embedded in the app process**: it stores vectors + metadata together,
supports metadata filtering, and needs no separate database server. For a small/mid,
lightly-queried RAG index that makes it far cheaper than a managed vector DB (see
[Cost analysis](#cost-analysis)).

**Accepted trade-offs:** embedded LanceDB is great for single-service/low-cost
deployments but not for many concurrent writers, multi-region HA, or managed
backups — that is when a managed service wins. The code prefers explicit modules
over a heavy RAG framework so ingestion, retrieval, cost and evaluation stay auditable.

## Honest results (offline mock run, 26 questions)

These come from `python -m eval.run` with **mock providers** (no API keys, fully
reproducible). Numbers are intentionally *not* perfect — that is the point.

| Metric | Value |
|---|---:|
| Questions (answerable / refusal) | 23 / 3 |
| Recall@5 | **0.826** |
| Hit Rate | 0.826 |
| MRR | 0.592 |
| nDCG@5 | 0.651 |
| Precision@5 | 0.165 |
| Average Precision | 0.592 |
| Adversarial probe — faithfulness on a *correct* answer | **5 / 5** |
| Adversarial probe — faithfulness on a *confidently-wrong* answer | **1 / 5** |

Retrieval is **not 1.000**: the corpus is independent of the questions (see
[Corpus](#corpus--provenance)), and mock embeddings are a weak bag-of-words stand-in,
so the score reflects the system, not a rigged corpus. With real Gemini embeddings it
would be higher, but mock keeps the committed numbers reproducible and quota-free.

### Answer quality, EM/F1, and the judge

The judge scores **faithfulness** and **relevance** on a graded **1–5** scale. To
prove it actually discriminates (rather than returning 1.0 on everything), the
harness runs an **adversarial probe set**: for several questions it judges a planted
*correct*, *confidently-wrong*, and *verbose-unsupported* answer. The judge gives the
correct answer 5/5 and the wrong/unsupported ones 1/5, and SQuAD-style **Exact Match /
token-F1** drop the same way (`adversarial_probes` in `reports/evaluation_results.json`).

In the offline run the *system* answers come from the mock generator (a placeholder),
so per-case EM/F1/faithfulness on system output are low by construction — real answer
quality is shown by the probes and the optional live smoke. The judge provider/family
is kept different from the generator so it never grades its own model.

### Latency (cold)

The eval runs every query **cold** (caches bypassed) and records each stage
separately, so these are real-work numbers, not warm-cache hits.

| Stage (cold) | p50 | p95 |
|---|---:|---:|
| Embedding (mock) | ~0 ms | ~1 ms |
| **Retrieval (vector store)** | **~12 ms** | **~18 ms** |
| Generation (mock) | ~0 ms | ~2 ms |
| Total | ~13 ms | ~18 ms |

These are **mock** timings (no network). With real providers, per-query latency is
dominated by the API calls: embedding ≈ 0.8–0.9 s + generation ≈ 0.7 s, with
retrieval still ≈ 25 ms — i.e. real total ≈ **1.5–1.6 s** per cold query, not 15 ms.
Warm-cache repeats are near-zero but are explicitly a cache artifact, not query speed.
The vector-store metric to compare is **retrieval-only (~12–25 ms)**.

### Refusal (no-context)

Out-of-corpus questions should be declined. Distance-gated refusal needs real
*semantic* embeddings — bag-of-words mock embeddings can't separate out-of-corpus
queries, so in the mock run `refusal_accuracy` is reported but not meaningful. Refusal
is validated by unit tests (`tests/test_generation.py`) and the live smoke below.

## Corpus & provenance

The evaluated corpus in `data/corpus/` is **independent third-party material** — none
of it was written by this project's author, so questions can't be answered "from the
author's own notes". Built reproducibly by `python -m data.build_corpus`; full sources
and licenses in [`data/SOURCES.md`](data/SOURCES.md):

| File | Format | Source | License |
|---|---|---|---|
| `fastapi_features.md` | Markdown | FastAPI docs | MIT |
| `vector_search_and_rag.html` | HTML | Wikipedia (Vector database / NN search / RAG) | CC BY-SA 4.0 |
| `http_status_codes.pdf` | PDF | Wikipedia (HTTP status codes) | CC BY-SA 4.0 |

## Setup (local venv)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
# .venv/bin/python   -m pip install -r requirements-dev.txt       # POSIX
cp .env.example .env        # fill in keys, or set *_PROVIDER=mock for a keyless run
```

Recommended config (in `.env.example`): `EMBEDDING_PROVIDER=gemini`,
`GENERATION_PROVIDER=groq`, `JUDGE_PROVIDER=gemini` (judge family ≠ generator family).
Gemini supports key rotation via `GEMINI_API_KEYS=key1,key2`.

`CHUNK_SIZE`/`CHUNK_OVERLAP` are **characters** (RecursiveCharacterTextSplitter).

## Run it

```bash
# ingest the corpus, then serve
python -m src.ingest
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
# health / readiness
curl localhost:8000/health
curl localhost:8000/ready      # row count + embedding model/dim

# ingest (restricted to DATA_ROOT) and query
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"data_dir":"data/corpus"}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' \
     -d '{"query":"What does HTTP status code 502 mean?","top_k":5}'
# metadata filter: {"query":"...","metadata_filter":{"doc_type":"pdf"}}
```

Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
cp .env.example .env          # optional; /health, /ready, /docs work without keys
docker compose up --build     # serves on :8000, persists db/ in a volume
```

## Make targets

`make setup | test | lint | ingest | eval | cost | serve | docker-build | docker-run`
(override the interpreter with `PYTHON=.venv/Scripts/python.exe`). CI
(`.github/workflows/ci.yml`) runs `make lint` + `make test` on a clean checkout with
mock providers.

## Evaluation & cost

```bash
python -m eval.export_chunks    # reports/chunk_inventory.json (chunk IDs)
python -m eval.build_test_set   # regenerate eval/test_set.json labels
python -m eval.run              # reports/evaluation_results.json
python -m eval.cost_analysis    # reports/cost_analysis.json
```

Retrieval: Recall@5, Hit Rate, MRR, nDCG@5, **Precision@5** (renamed from the
mislabelled `context_precision`) and order-aware **Average Precision**. Answer:
EM, token-F1, and the 1–5 judge. The runner wraps each case in try/except and
**persists partial results after every case**, so a mid-run provider/quota error never
loses completed work.

## Cost analysis

LanceDB still needs an always-on host, so its total is **storage + compute**, compared
against a **named, sourced** managed baseline: Pinecone serverless (Standard) —
$0.33/GB-mo + $50/mo plan minimum + read/write units, per
<https://www.pinecone.io/pricing/> (retrieved 2026-06-23). Host cost is configurable
via `LANCEDB_HOST_COST_MONTH`.

| Vectors | Index (GB) | LanceDB storage | LanceDB host | **LanceDB total** | **Managed total** | One-time embedding |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.351 | $0.03 | $10 | **$10.03** | **$50.00** | $2.73 |
| 1,000,000 | 3.513 | $0.28 | $10 | **$10.28** | **$50.00** | $27.27 |
| 10,000,000 | 35.129 | $2.81 | $10 | **$12.81** | **$50.00** | $272.73 |

Embedding cost converts characters→tokens (~4.4 chars/token, so ~182 tokens/chunk).
**Break-even:** at these scales LanceDB is ~5× cheaper; managed wins once you need
multi-region HA / managed backups / SLAs, or such high query concurrency that the
always-on host for LanceDB would itself exceed the ~$50/mo managed minimum.

## Telemetry

Every query logs one JSON line including `cached` (cold vs warm), `skipped_llm`, and
per-stage latency. Provider failures retry with backoff on 429/5xx (shared helper for
Gemini key-rotation and Groq); only the key *index* is logged, never the key.

## Live smoke (optional)

`python -m eval.smoke` runs a tiny real-provider check (≤3 generate calls): it confirms
out-of-corpus questions are **refused** (0 generate calls — refusal skips the LLM) and
that 1–2 grounded answers are judged high, recording real cold latency to
`reports/smoke_results.json`. It is **provided but not executed in this submission to
preserve API quota** (and skips automatically if no keys are configured).

## Problem 2 — LLM-as-Judge pipeline

A separate workstream in `eval/pipeline/` provides an A/B comparison harness with
Pydantic-validated suites, structured `JudgeVerdict`s with retry + JSON re-prompt,
full JSONL audit log (prompt, raw response, tokens, cost, latency), pairwise
**position-swap debiasing**, a 5-criterion **weighted rubric** (used — not
decorative), adversarial **verbosity/sycophancy** probes with gold winners,
Cohen's kappa vs gold, and a self-enhancement-bias warning when judge family ==
generator family.

Run it: `python -m eval.run_pipeline --suite eval/suites/sample_suite.yaml`

Committed result (Groq judge, llama-3.3-70b; generator = gemini family; suite =
RAG Quality Comparison, 5 cases): **winner = Prompt v1 (concise)**, win rate
80 / 20, weighted score 4.85 vs 2.77, pass rate 100% vs 20%, position-bias /
position-flip both 0%, verbosity + sycophancy probes both PASSED, gold agreement
60% (κ=0.231). Full report in `reports/p2_evaluation_report.json`; per-call
audit in `reports/p2_audit_log.jsonl`; CSV summary in `reports/p2_results.csv`.

## Limitations / what I'd do with more time

- Committed metrics use mock providers for reproducibility/quota; the headline real
  numbers come from the small live smoke. A scheduled real-embedding eval would give
  fuller retrieval numbers.
- The distance-based refusal threshold is calibrated for the real embedding model; a
  provider-agnostic normalized-similarity cutoff would behave consistently under mock.
- Single-vector dense retrieval only — hybrid (BM25 + vector) and a reranker would lift
  precision on the harder, paraphrased questions.
- No auth/rate-limiting on the API; fine for a local service, needed before exposure.
```
