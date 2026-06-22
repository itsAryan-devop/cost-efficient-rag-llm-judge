# Problem 1 Rubric Evidence

This file maps the submitted artifacts to the Problem 1 scoring rubric.

## Correctness And Ingestion - 20 pts

Evidence:

- `src/ingestion.py` supports PDF, HTML, Markdown, and text files.
- `src/storage.py` stores vectors, chunk text, embedding model, embedding dimensionality, stable document ID, content hash, source file, document type, chunk index, chunk size, and chunk overlap.
- `tests/test_storage.py` verifies idempotent LanceDB upsert behavior and stale-chunk replacement when a document changes.
- `data/assignment.pdf`, `data/cost_analysis_notes.html`, and `data/rag_architecture_notes.md` prove all required ingestion formats are exercised.

## Retrieval Evaluation - 20 pts

Evidence:

- `eval/test_set.json` contains 15 fixed evaluation questions with labeled relevant chunk IDs.
- `eval/ir_metrics.py` computes Recall@k, Hit Rate, MRR, nDCG@k, and context precision.
- `reports/evaluation_results.json` contains per-question retrieval results and aggregate metrics.

Latest aggregate retrieval metrics:

- Recall@5: 0.967
- Hit Rate: 1.000
- MRR: 0.913
- nDCG@5: 0.916
- Context Precision: 0.293

## Answer Evaluation - 20 pts

Evidence:

- `eval/llm_judge.py` scores faithfulness/groundedness and answer relevance.
- `reports/evaluation_results.json` includes faithfulness and relevance per case.

Latest aggregate answer metrics:

- Faithfulness: 1.000
- Answer relevance: 1.000

## Cost Analysis - 20 pts

Evidence:

- `eval/cost_analysis.py` generates a reproducible cost analysis.
- `reports/cost_analysis.json` includes assumptions and cost table.
- README includes the same cost table and discussion of when to switch back to managed infrastructure.

## Engineering And Clarity - 20 pts

Evidence:

- FastAPI HTTP service in `src/api.py`.
- Environment-only config in `src/config.py` and `.env.example`.
- Gemini key rotation in `src/gemini_client.py`.
- Structured JSON telemetry in `src/logger.py`.
- Cache-backed embeddings/generation/judging to reduce API usage.
- No-context fallback using `MIN_RELEVANCE_SCORE`.
- `README.md` includes setup, API usage, evaluation workflow, cost analysis, and discussion.
- `tests/` currently passes with 21 tests.
