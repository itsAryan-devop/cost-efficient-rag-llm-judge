"""Tiny real-provider smoke test (<=3 generate calls).

Confirms the real pipeline end-to-end: out-of-corpus questions are refused
(refusal skips the LLM, so 0 generate calls), and a couple of grounded answers
are generated and judged with real cold latency. Writes
``reports/smoke_results.json``. Skips cleanly if no API keys are configured.

Run manually (it costs quota):
    python -m eval.smoke
"""

from __future__ import annotations

import json
import os
import time

from src.config import settings
from src.gemini_client import get_gemini_api_keys

# Use a separate DB so the mock-embedded index is never overwritten.
os.environ.setdefault("DB_PATH", "db/smoke_lancedb")
os.environ.setdefault("CACHE_PATH", "cache/smoke_diskcache")

ANSWERABLE = [
    "What does HTTP status code 502 Bad Gateway mean?",
    "Which two open standards is FastAPI based on?",
]
REFUSAL = ["What is the capital of France?"]


def _have_keys() -> bool:
    return bool(get_gemini_api_keys()) and bool(settings.groq_api_key)


def main() -> None:
    if settings.embedding_provider == "mock" or not _have_keys():
        print(
            "Smoke skipped: set real providers + GEMINI/GROQ keys to run " "(skipped here to preserve quota)."
        )
        return

    # Import after env is set so caches/DB point at the smoke locations.
    from eval.llm_judge import evaluate_answer
    from eval.run import _judge_context
    from src.embedding import get_embedding
    from src.generation import generate_answer
    from src.ingest import run_ingest
    from src.storage import search

    run_ingest(settings.data_root)  # real embeddings into the smoke DB

    results = []
    generate_calls = 0
    for query, expect_refusal in [(q, False) for q in ANSWERABLE] + [(q, True) for q in REFUSAL]:
        t0 = time.time()
        qv = get_embedding(query, input_type="query", use_cache=False)
        emb_ms = (time.time() - t0) * 1000
        t1 = time.time()
        sources = search(qv, top_k=5)
        ret_ms = (time.time() - t1) * 1000
        t2 = time.time()
        gen = generate_answer(query, sources, use_cache=False)
        gen_ms = (time.time() - t2) * 1000
        if not gen.skipped_llm:
            generate_calls += 1

        record = {
            "query": query,
            "expected_refusal": expect_refusal,
            "refused": gen.skipped_llm,
            "answer": gen.answer,
            "cold_latency_ms": {
                "embedding": round(emb_ms, 1),
                "retrieval": round(ret_ms, 1),
                "generation": round(gen_ms, 1),
                "total": round(emb_ms + ret_ms + gen_ms, 1),
            },
        }
        if not expect_refusal and not gen.skipped_llm:
            judge = evaluate_answer(query, _judge_context(gen.answer, sources), gen.answer)
            record["faithfulness_1to5"] = judge.faithfulness_score
            record["relevance_1to5"] = judge.relevance_score
        results.append(record)

    report = {"generate_calls": generate_calls, "results": results}
    os.makedirs(settings.reports_path, exist_ok=True)
    out = os.path.join(settings.reports_path, "smoke_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Smoke done ({generate_calls} generate calls). Saved {out}")


if __name__ == "__main__":
    main()
