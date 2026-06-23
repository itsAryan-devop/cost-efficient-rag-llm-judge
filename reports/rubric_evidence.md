# Problem 1 Rubric Evidence

Maps submitted artifacts to the Problem 1 scoring rubric. All metrics below are
from the reproducible offline mock run (`python -m eval.run`); see
[`README.md`](../README.md) for the honest-results discussion.

## Correctness And Ingestion — 20 pts

- `src/ingestion.py` parses PDF / HTML / Markdown / text; `src/ingest.py` is the
  shared embed+upsert pipeline used by the API and CLI.
- `src/storage.py` stores vector, text, embedding model + dimension, stable
  document ID, content hash, source file, doc type, chunk index/size/overlap.
- `tests/test_storage.py` — idempotent upsert + stale-chunk replacement.
- `tests/test_api.py` — hermetic ingest+query on a temp DB (`mock_corpus_db`).
- Corpus: `data/corpus/` (PDF + HTML + Markdown), provenance in `data/SOURCES.md`.

## Retrieval Evaluation — 20 pts

- `eval/ir_metrics.py`: Recall@k, Hit Rate, MRR, nDCG@k, **Precision@k** (renamed
  from the mislabelled `context_precision`) and order-aware **Average Precision**;
  unit-tested in `tests/test_ir_metrics.py`.
- `eval/test_set.json`: 26 honestly-labelled questions (23 answerable incl.
  hard/multi-chunk + paraphrased, 3 out-of-corpus refusals), built by
  `eval/build_test_set.py` against real chunk IDs.
- `reports/evaluation_results.json` — **Recall@5 = 0.826 (not 1.0)**, MRR 0.592,
  nDCG@5 0.651, Precision@5 0.165.

## Answer Evaluation — 20 pts

- `eval/llm_judge.py`: graded **1–5** faithfulness + relevance with rubric anchors;
  the mock judge is a deterministic lexical-grounding heuristic.
- **Discrimination proven**: `adversarial_probes` in the report and
  `tests/test_llm_judge.py` show a *correct* answer scored 5/5 and a
  *confidently-wrong* / *verbose-unsupported* answer scored 1/5.
- `eval/text_metrics.py`: SQuAD-style **Exact Match** + **token-F1** vs
  `reference_answer`; unit-tested; `mean_exact_match`/`mean_token_f1` in the summary.
- Judge family kept ≠ generator family; rationale + raw response logged per case.

## Cost Analysis — 20 pts

- `eval/cost_analysis.py` + `reports/cost_analysis.json`: LanceDB **storage + host**
  vs a **named, sourced** managed baseline (Pinecone serverless, pricing dated
  2026-06-23), no unused assumption fields, chars→tokens reconciled, break-even note.

## Engineering And Clarity — 20 pts

- FastAPI service (`src/api.py`) with `/health`, `/ready` (row count + embedding
  info), `/ingest`, `/query`; structured JSON telemetry incl. cold-vs-cached.
- Shared 429/5xx retry helper (`src/retry.py`) for Gemini key-rotation and Groq.
- Fault-tolerant eval: per-case try/except + incremental partial-result persistence.
- Pinned `requirements.txt`, `pyproject.toml` (ruff/black), `Makefile`,
  `.github/workflows/ci.yml`, `Dockerfile` + `docker-compose.yml`, `render.yaml`.
- `rm -rf db cache && pytest` is green on a clean checkout.
