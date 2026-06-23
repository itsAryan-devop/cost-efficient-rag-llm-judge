# CHANGES — Problem 1 hardening (reviewer defects → fixes)

Scope: Problem 1 only. Each item maps to a reviewer defect.

## P0 — highest-impact

1. **Fake-perfect retrieval / self-referential corpus** → Removed the author-written
   crib notes + `assignment.pdf`. Added an independent, permissively-licensed corpus
   in `data/corpus/` (FastAPI MIT Markdown, Wikipedia CC BY-SA HTML + PDF), built
   reproducibly by `data/build_corpus.py` with provenance in `data/SOURCES.md`.
   Rebuilt `eval/test_set.json` to 26 honestly-labelled questions (23 answerable incl.
   hard/multi-chunk, 3 out-of-corpus refusals). **Recall@5 is now 0.826, not 1.000.**
2. **Storage-only cost comparison / unused assumption** → `eval/cost_analysis.py` now
   reports LanceDB **storage + host** vs a named, sourced managed baseline (Pinecone
   serverless, dated 2026-06-23). Removed unused fields; reconciled chars→tokens; added
   a break-even sentence.
3. **Binary judge that never failed / no EM/F1** → Graded **1–5** judge with rubric
   anchors and a deterministic discriminating mock judge. Added an **adversarial probe
   set** proving a planted wrong answer scores 1/5 vs 5/5 for the correct one. Added
   `eval/text_metrics.py` (SQuAD EM + token-F1), unit-tested, with `mean_em`/`mean_f1`.
4. **Dishonest 15 ms latency** → `eval/run.py` runs every query **cold** (caches
   bypassed) and records embedding/retrieval/generation/total separately; README leads
   with cold numbers and labels warm-cache as such.

## P1 — robustness / hygiene

5. **Groq had no retry; one error aborted the eval** → Shared 429/5xx backoff helper
   (`src/retry.py`) for Gemini + Groq; per-case try/except with incremental
   partial-result persistence; failed-case summary.
6. **Test depended on developer-local db/** → `mock_corpus_db` fixture ingests a temp
   corpus; `rm -rf db cache && pytest` is fully green.
7. **Mislabelled `context_precision`** → Renamed to `precision_at_k` everywhere; added
   order-aware `average_precision`; unit-tested.
8. **Config drift** → One `JUDGE_PROVIDER=gemini` across config/.env.example/README;
   stated CHUNK_SIZE/OVERLAP are characters; reconciled the token assumption.
9. **Deps / hygiene** → Pinned `requirements.txt`; removed unused `markdown`; split
   `requirements-dev.txt`; added `.dockerignore`; clean UTF-8 `.env.example`.

## P2 — productionization

10. `Dockerfile` (multi-stage, non-root) + `docker-compose.yml`.
11. `Makefile` + `pyproject.toml` ruff/black config (`make lint` clean).
12. `.github/workflows/ci.yml` — lint + test on a clean checkout, mock providers.
13. `render.yaml` + README Deploy section.
14. `/ready` endpoint (row count + embedding model/dim); cold-vs-cached telemetry.

## Notes

- Committed metrics are from the **mock** offline run (reproducible, quota-free).
  `eval/smoke.py` provides a ≤3-generate-call real smoke; not executed in this
  submission to preserve API quota.
- Problem 2 (LLM-as-Judge) files under `eval/pipeline/` are a separate workstream and
  were intentionally left untouched.
