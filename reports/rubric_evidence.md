# Problem 1 Rubric Evidence

This file maps the submitted artifacts to the Problem 1 scoring rubric.

## Correctness And Ingestion - 20 pts

Evidence:

- `src/ingestion.py` parses PDF, HTML, Markdown, and text files.
- `python -m src.ingestion` provides the CLI ingest path.
- `src/storage.py` stores vectors, text, embedding model, embedding dimension, stable document ID, content hash, source file, doc type, chunk index, chunk size, and chunk overlap.
- `src/api.py` exposes `/health`, `/ready`, `/ingest`, and `/query`.
- `src/api.py` restricts API-triggered ingestion to `DATA_ROOT`.
- `tests/test_storage.py` verifies idempotent upsert and stale-chunk replacement.
- `tests/test_api.py` verifies API ingest/query behavior using a hermetic temp DB.
- Corpus files live in `data/corpus/`; provenance is documented in `data/SOURCES.md`.

## Retrieval Evaluation - 20 pts

Evidence:

- `eval/test_set.json` contains 26 labeled questions: 23 answerable and 3 out-of-corpus refusal cases.
- `eval/build_test_set.py` builds labels against real chunk IDs.
- `eval/ir_metrics.py` computes Recall@k, Hit Rate, MRR, nDCG@k, Precision@k, and Average Precision.
- `tests/test_ir_metrics.py` unit-tests metric calculations.
- `reports/evaluation_results.json` contains per-question retrieved IDs and aggregate retrieval metrics.

Latest offline aggregate retrieval metrics:

- Recall@5: 0.826
- Hit Rate: 0.826
- MRR: 0.592
- nDCG@5: 0.651
- Precision@5: 0.165
- Average Precision: 0.592

## Answer Evaluation - 20 pts

Evidence:

- `eval/llm_judge.py` scores faithfulness and relevance on a 1-5 rubric with rationales.
- The judge supports real Gemini/Groq providers and an offline deterministic mock mode.
- `reports/evaluation_results.json` includes EM, token-F1, faithfulness, relevance, rationales, and raw judge outputs.
- Offline refusal accuracy is 1.0 on the 3 out-of-corpus cases.
- Adversarial probes prove the judge discriminates:
  - Correct grounded answers score 5/5.
  - Confidently wrong answers score 1/5.
- `reports/smoke_results.json` provides a tiny real-provider smoke run:
  - Two grounded answers scored faithfulness 5/5 and relevance 5/5.
  - One out-of-corpus query refused correctly.

## Cost Analysis - 20 pts

Evidence:

- `eval/cost_analysis.py` generates the cost report.
- `reports/cost_analysis.json` compares LanceDB storage plus host cost against Pinecone serverless.
- The report states vector dimension, bytes per float, metadata overhead, storage cost, host cost, embedding cost, and scale assumptions.
- The cost table covers 100K, 1M, and 10M vectors.

## Engineering And Clarity - 20 pts

Evidence:

- Environment-only config in `src/config.py` and `.env.example`.
- Gemini key rotation in `src/gemini_client.py`.
- Groq key rotation and shared retry/backoff in `src/retry.py`.
- Gemini-primary/Groq-fallback generation in `src/generation.py`.
- Gemini-primary/Groq-fallback judging in `eval/llm_judge.py`.
- Structured JSON telemetry in `src/logger.py`.
- Cache-backed embeddings/generation/judging to reduce API usage.
- No-context fallback using `MAX_RETRIEVAL_DISTANCE` plus prompt-level refusal.
- Dockerfile, docker-compose, CI workflow, Makefile, pinned requirements, and lint config.
- `README.md` includes setup, API usage, evaluation workflow, cost analysis, evidence summary, and limitations.
- Current test suite: 73 passed.
