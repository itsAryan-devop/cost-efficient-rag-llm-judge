# HANDOFF — Make Problem 1 (Cost-Efficient RAG) production-grade in ONE pass

You are Claude Code working in this repo (`cost efficient rag project`). A senior reviewer graded the
current state **69/100** and wrote the exact defects below. Your job: **fix all of them, end-to-end,
autonomously, in a single run**, then leave the repo green, reproducible, containerized, and
deployable. Scope = **Problem 1 only** (Part 2 / LLM-as-Judge is handled separately — do NOT build it).

Hard constraints:
- The owner is on a deadline and low on API quota. **Do not burn quota.** Develop and test with
  `*_PROVIDER=mock`. Make at most ONE tiny real-provider smoke run at the very end (≤3 generate calls),
  and only if quota is available; otherwise skip and say so.
- Never commit secrets. `.env` stays gitignored. Only edit `.env.example`.
- Everything must pass on a **clean checkout** (fresh clone, no pre-existing `db/`), in CI, and in Docker.
- Make small, logically-grouped git commits with clear messages as you go.

Known-good facts (don't waste time re-checking): `gemini-embedding-2` is a real model and returns
768-dim vectors; ingestion + idempotency + no-context refusal + `doc_type` metadata filter already work
live; 30/31 tests pass once a DB exists. Keep all of that working — don't regress it.

---

## P0 — The four things that cost the most marks (do these first)

### 1. Replace the self-referential corpus with an independent one
Problem: the corpus is `assignment.pdf` + three notes files the candidate wrote, and the eval questions
are answered nearly verbatim by those notes → Recall@5 = Hit Rate = 1.000 is fake-easy. (With mock
embeddings recall drops to 0.86, proving the corpus, not the system, produced the perfect score.)

Do:
- Add a real, neutral corpus under `data/` that the candidate did NOT write the answers to. Use a few
  public, redistributable technical docs (e.g. download 3–6 pages/sections of well-known docs:
  FastAPI, LanceDB, HTTP/REST, vector-search concepts — pick sources with permissive terms, and record
  the source URL + license in `data/SOURCES.md`). Mix formats: ≥1 PDF, ≥1 HTML, ≥1 Markdown.
- Remove the self-authored "crib notes" (`rag_architecture_notes.md`, `sample_rag_notes.md`,
  `cost_analysis_notes.html`) from the evaluated corpus. You may keep `assignment.pdf` only if you also
  have genuinely independent docs; it should not dominate.
- Rebuild `eval/test_set.json` with **20–30** questions (not the bare 15) whose answers live in the new
  corpus, with honestly-labeled `relevant_chunk_ids` (use `eval/export_chunks.py` to get IDs) and gold
  `reference_answer`s. Include a few **genuinely hard** questions (multi-chunk, paraphrased, not keyword
  matches) and **2–3 out-of-corpus "should-refuse" questions** with `relevant_chunk_ids: []`.
- Re-run ingestion so IDs match the new corpus.
- Acceptance: retrieval metrics are computed on the new corpus and are **not** all 1.000; the report
  states corpus provenance.

### 2. Make the cost comparison apples-to-apples
Problem: LanceDB cost in the table is **storage only** ($0.03–$2.81/mo). Embedded LanceDB still needs an
always-on host. `cost_analysis.py` even defines `existing_app_server_cost_month = 10.0` and never uses it.
Managed baseline ($70 flat, +$10/M) is unsourced.

Do (`eval/cost_analysis.py` + README table):
- Add a **compute/host line** to the LanceDB total (use the already-defined app-server cost; make it
  configurable). Show LanceDB total = storage + host, so it's comparable to managed (compute+storage).
- Replace the hand-wavy managed baseline with **at least one named, cited managed option** (e.g. Pinecone
  serverless or Qdrant Cloud) using their **published pricing as of the report date**, with the pricing
  assumption and date written in the report. Keep it conservative and clearly labeled "estimate."
- Keep the three scales (100K/1M/10M). Add a one-line **break-even / when-managed-wins** sentence.
- Acceptance: the table has LanceDB(storage+compute) vs Managed(named, sourced); no unused assumption
  fields; README and `reports/cost_analysis.json` agree.

### 3. Make answer evaluation actually discriminate + add EM/F1
Problem: judge is binary 0/1 and returns 1.0 on all cases (even the mock judge returns 1) → never shown
to catch a bad answer. And every test case has a gold `reference_answer`, but EM/F1 is never computed
(the prompt says compute it when gold answers exist).

Do (`eval/llm_judge.py`, `eval/run.py`, new `eval/text_metrics.py`):
- Move faithfulness/relevance to a **graded scale** (e.g. 1–5) with rubric anchors, OR keep binary but
  ADD a third metric and prove discrimination. Either way you MUST demonstrate the judge returning a
  **low score on a deliberately wrong answer**.
- Add an **adversarial probe set** in the harness: for ≥2 cases, also judge a planted
  "confidently-wrong" answer and a "verbose-but-unsupported" answer; assert/report that the judge scores
  them low. Save these in the results file.
- Implement **EM and token-level F1** vs `reference_answer` (normalize: lowercase, strip punctuation/
  articles, whitespace — SQuAD-style) in `eval/text_metrics.py`, unit-test it, and add `mean_em` /
  `mean_f1` to the aggregate summary.
- Keep judge model family ≠ generator family (good already). Keep rationale + raw response logged.
- Acceptance: summary includes EM, F1, and judge scores that are NOT uniformly perfect; an adversarial
  case shows a low judge score in `reports/evaluation_results.json`.

### 4. Honest latency reporting
Problem: README leads with p50 15ms / p95 16ms — those are **warm-cache cache hits**
(`embedding_latency_ms`/`generation_latency_ms` = 0.0). Real per-query latency measured ≈ **1.6s**
(embed ~0.8–0.9s + generate ~0.7s; retrieval ~25ms).

Do:
- In `eval/run.py`, run each query **cold once** (cache cleared / bypassed for the timing pass) and
  record cold p50/p95 for **embedding, retrieval, generation, and total** separately.
- README must show **cold** numbers as the headline, with retrieval-only called out as the
  vector-store metric, and clearly label any warm-cache numbers as such.
- Acceptance: README latency section distinguishes cold vs warm and the cold totals are realistic
  (hundreds of ms–seconds), not 15ms.

---

## P1 — Robustness, correctness, hygiene (all required for "production-level")

### 5. Provider resilience parity + fault-tolerant eval
Problem: only Gemini has retry+rotation; Groq path has none — a single 429 became a 502. And
`run_evaluation()` has no per-case try/except, so one error aborts the whole run and saves nothing
(reproduced on both Groq daily-cap and Gemini free-tier 20/day cap).

Do:
- Give the Groq path (`src/generation.py`, `eval/llm_judge.py`) the same treatment: retry with backoff on
  429/5xx, clear error surface. Factor a small shared retry helper so Gemini and Groq behave consistently.
- In `eval/run.py`: wrap **each case** in try/except, record per-case error + continue, and **persist
  partial results** incrementally (write after each case or in a finally block) so a mid-run quota error
  never loses completed work. Print a summary of failed cases.
- Acceptance: killing one provider mid-eval still yields a saved report with partial results and logged
  errors.

### 6. Hermetic tests (must pass on a clean checkout)
Problem: `test_query_response_model_excludes_vectors` depends on a pre-populated `db/lancedb`; on a fresh
clone it fails (`assert []`).

Do:
- Add a pytest fixture that ingests a tiny fixture corpus into a **temp DB** (mock embeddings) before the
  query tests, or have the test ingest its own data. No test may depend on developer-local state.
- Add a CI test that simulates a clean checkout: fresh temp dirs for `DB_PATH`/`CACHE_PATH`, mock
  providers, full `pytest` green.
- Acceptance: `rm -rf db cache && pytest -q` is **all green** (currently 1 fails).

### 7. Fix the mislabeled metric
`context_precision` is actually precision@k. Either rename it to `precision_at_k` everywhere
(report keys, README, tests) **or** implement the real order-aware context precision (average precision
over the ranked relevant hits) and label it correctly. Pick one and be consistent. Unit-test it.

### 8. Config + docs consistency
- Resolve the drift: `.env.example`, README "recommended config", and committed report disagree on
  `JUDGE_PROVIDER` (groq vs gemini). Choose ONE default, make `.env.example`, README, and config defaults
  all agree, and regenerate the committed report with that config.
- State explicitly that `CHUNK_SIZE=1000` / `CHUNK_OVERLAP=200` are **characters** (RecursiveCharacter
  splitter), and reconcile the "180 tokens/chunk" cost assumption (either switch the cost model to a
  chars→tokens conversion or relabel; don't leave the unit mismatch unstated).

### 9. Dependency + repo hygiene
- **Pin every dependency** in `requirements.txt` to known-good versions (use what's installed/working).
- Remove the unused `markdown` dependency (it's never imported) — or actually use it to render MD→text.
- Add a `.dockerignore`. Confirm `.env` is gitignored (it is) and never gets committed.
- Strip the BOM from `.env.example` if present; keep files UTF-8 no-BOM.

---

## P2 — Productionization (this is what turns "demoable" into "hire him")

### 10. Containerize
- `Dockerfile` (slim Python base, non-root user, install pinned deps, copy `src/`, expose 8000,
  `CMD uvicorn src.api:app --host 0.0.0.0 --port 8000`). Multi-stage if it keeps the image small.
- `docker-compose.yml` for one-command local run (`docker compose up`), env via `.env`,
  volume-mount `data/` and persist `db/`.
- Acceptance: `docker compose up` serves `/health` 200 and `/docs`; a documented `curl` ingest+query works.

### 11. One-command DX
- `Makefile` (or `justfile`) targets: `setup`, `test`, `ingest`, `eval`, `cost`, `serve`, `docker-build`,
  `docker-run`, `lint`. Keep them thin wrappers over the existing module entrypoints.
- Add `ruff` (or `flake8`) + `black --check` and wire into `make lint`.

### 12. CI
- Add `.github/workflows/ci.yml`: install pinned deps, `make lint`, `make test` with mock providers on a
  clean checkout (fresh temp DB/cache). Must be green. No secrets needed (mock only).

### 13. Optional deploy (only if time remains after everything above is green)
- Add `render.yaml` (Render web service) **and** a short README "Deploy" section. Use env vars for keys
  (never hardcode). This is a nice-to-have signal; do NOT let it block P0/P1.

### 14. Health/readiness + minimal observability polish
- Keep `/health`; add a `/ready` that confirms the DB table is reachable and reports row count + embedding
  model/dim. Keep the existing structured JSON telemetry; ensure every query logs cold vs cached.

---

## Final README rewrite (honest + reproducible)
Rewrite `README.md` so a stranger can reproduce everything:
- 30-second "what + why LanceDB" + accepted trade-offs (keep the good existing discussion).
- Exact run steps for: local venv, Docker, ingest, query (with `curl` examples), eval, cost.
- **Honest** results: corpus provenance, retrieval metrics (not all 1.0), answer metrics incl. EM/F1 and
  the adversarial-probe result, cold latency p50/p95, and the corrected cost table with sources.
- A short "Limitations / what I'd do with more time" section (shows judgment).
- Update `reports/rubric_evidence.md` to point at the new artifacts.
- Keep git history clean and incremental.

---

## Definition of Done — verify ALL before you stop (run this yourself)
1. `rm -rf db cache && EMBEDDING_PROVIDER=mock GENERATION_PROVIDER=mock JUDGE_PROVIDER=mock pytest -q`
   → **all green**, including the previously-failing query test.
2. `make lint` → clean. CI workflow present and would pass (mock-only).
3. Fresh ingest on the **new independent corpus** works; re-ingest is idempotent (row count stable).
4. `python -m eval.run` (mock ok for logic) produces a report with: retrieval metrics **not all 1.0**,
   EM + F1 present, judge scores that include a **low score on the planted wrong answer**, and cold
   latency split by stage. Partial-save works if a case errors.
5. `python -m eval.cost_analysis` → table compares **LanceDB(storage+compute)** vs **named/sourced
   managed**; no unused assumption fields.
6. `docker compose up` → `/health` 200, `/docs` loads, documented curl ingest+query succeeds.
7. README reflects real numbers (no 15ms headline; no all-1.0 retrieval) and corpus sources.
8. No secrets committed; `.env` gitignored; deps pinned; `markdown` dep resolved.
9. (Optional, only if quota available) one tiny real-provider smoke (≤3 generate calls) using Gemini key
   rotation; otherwise note it was skipped to preserve quota.

Work top-down (P0 → P1 → P2). If you must drop anything for time, drop P2 #13 first, never P0. Commit as
you complete each numbered item. Leave a short `CHANGES.md` summarizing what you fixed mapped to the
reviewer's defects.
