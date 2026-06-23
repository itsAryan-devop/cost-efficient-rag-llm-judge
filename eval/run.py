"""Evaluation harness for the cost-efficient RAG service.

Produces ``reports/evaluation_results.json`` with:

* Retrieval metrics (Recall@5, Hit Rate, MRR, nDCG@5, Precision@5, Average
  Precision) over the *answerable* questions.
* Answer metrics: SQuAD-style EM / token-F1 vs the gold ``reference_answer`` and
  a graded (1-5) faithfulness/relevance judge.
* An **adversarial probe set** that judges planted correct / confidently-wrong /
  verbose-unsupported answers to demonstrate the judge (and EM/F1) actually
  discriminate.
* Refusal accuracy over the out-of-corpus questions.
* **Cold** per-stage latency (embedding, retrieval, generation, total): every
  query is run once with the cache bypassed so the numbers reflect real work,
  not warm-cache hits.

Each case is isolated in try/except and results are persisted after every case,
so a mid-run provider error (e.g. a quota cap) never loses completed work.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone

from eval.ir_metrics import (
    average_precision,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from eval.llm_judge import evaluate_answer
from eval.text_metrics import exact_match, token_f1
from src.config import settings
from src.embedding import get_embedding
from src.generation import NO_CONTEXT_MESSAGE, generate_answer
from src.logger import log_event
from src.storage import search

TOP_K = 5

# Planted answers used to prove the judge and EM/F1 discriminate. Context text is
# quoted from the committed corpus so the probe is self-contained and offline.
ADVERSARIAL_PROBES = [
    {
        "query": "Which two open standards is FastAPI based on?",
        "context": (
            "Based on open standards: OpenAPI for API creation, including declarations of path "
            "operations, parameters, request bodies, security, etc. Automatic data model "
            "documentation with JSON Schema (as OpenAPI itself is based on JSON Schema)."
        ),
        "reference_answer": "FastAPI is based on OpenAPI and JSON Schema.",
        "answers": {
            "correct": "FastAPI is based on OpenAPI and JSON Schema.",
            "confidently_wrong": "FastAPI is based on the Flask and Django web frameworks.",
            "verbose_unsupported": (
                "FastAPI is an extraordinarily powerful, synergistic platform that leverages "
                "cutting-edge quantum paradigms and blockchain abstractions to revolutionize "
                "enterprise-grade workflows at planetary scale."
            ),
        },
    },
    {
        "query": "What does HTTP status code 502 Bad Gateway mean?",
        "context": (
            "502 Bad Gateway The server was acting as a gateway or proxy and received an invalid "
            "response from the upstream server."
        ),
        "reference_answer": (
            "The server, acting as a gateway or proxy, received an invalid response from the "
            "upstream server."
        ),
        "answers": {
            "correct": "It means a server acting as a gateway or proxy received an invalid response from the upstream server.",
            "confidently_wrong": "It means the client submitted invalid login credentials and must authenticate again.",
            "verbose_unsupported": (
                "The 502 code is a profound meditation on the ephemeral nature of distributed "
                "consensus, entropy, and the cosmic ballet of asynchronous existence."
            ),
        },
    },
]


def _mean(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _judge_context(answer: str, sources: list[dict]) -> str:
    cited_ids = set(re.findall(r"\b[a-f0-9]{64}\b", answer))
    selected_sources = [s for s in sources if s["id"] in cited_ids] or sources
    return "\n\n".join(
        f'<chunk id="{s["id"]}" source="{s.get("source_file", "")}">\n{s["text"]}\n</chunk>'
        for s in selected_sources
    )


def _run_cold_query(query: str):
    """Embed -> retrieve -> generate once with the cache bypassed (cold timings)."""
    t0 = time.time()
    query_vector = get_embedding(query, input_type="query", use_cache=False)
    embedding_ms = (time.time() - t0) * 1000

    t1 = time.time()
    sources = search(query_vector, top_k=TOP_K)
    retrieval_ms = (time.time() - t1) * 1000

    t2 = time.time()
    generation = generate_answer(query, sources, use_cache=False)
    generation_ms = (time.time() - t2) * 1000

    total_ms = embedding_ms + retrieval_ms + generation_ms
    log_event(
        "eval_query",
        query=query,
        cached=False,
        chunk_count=len(sources),
        skipped_llm=generation.skipped_llm,
        embedding_latency_ms=round(embedding_ms, 2),
        retrieval_latency_ms=round(retrieval_ms, 2),
        generation_latency_ms=round(generation_ms, 2),
    )
    return (
        sources,
        generation,
        {
            "embedding_ms": embedding_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    )


def _evaluate_case(item: dict) -> dict:
    query = item["query"]
    relevant_ids = item.get("relevant_chunk_ids", [])
    reference = item.get("reference_answer", "")
    expected_refusal = item.get("expected_refusal", False)

    sources, generation, lat = _run_cold_query(query)
    retrieved_ids = [s["id"] for s in sources]
    answer = generation.answer
    context_str = _judge_context(answer, sources)

    judge = evaluate_answer(query, context_str, answer)
    em = exact_match(answer, reference) if reference else None
    f1 = token_f1(answer, reference) if reference else None

    record = {
        "query": query,
        "expected_refusal": expected_refusal,
        "difficulty": item.get("difficulty"),
        "reference_answer": reference,
        "answer": answer,
        "skipped_llm": generation.skipped_llm,
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": relevant_ids,
        "exact_match": em,
        "token_f1": f1,
        "faithfulness_1to5": judge.faithfulness_score,
        "faithfulness_rationale": judge.faithfulness_rationale,
        "relevance_1to5": judge.relevance_score,
        "relevance_rationale": judge.relevance_rationale,
        "judge_raw_response": judge.raw_response,
        "judge_provider": judge.provider,
        "judge_model": judge.model,
        "embedding_latency_ms": round(lat["embedding_ms"], 2),
        "retrieval_latency_ms": round(lat["retrieval_ms"], 2),
        "generation_latency_ms": round(lat["generation_ms"], 2),
        "total_latency_ms": round(lat["total_ms"], 2),
        "token_usage": generation.token_usage,
        "error": None,
    }

    if expected_refusal:
        refused = generation.skipped_llm or answer.strip() == NO_CONTEXT_MESSAGE
        record["refused"] = refused
        record["refusal_correct"] = refused
    else:
        record["recall@5"] = recall_at_k(retrieved_ids, relevant_ids, k=TOP_K)
        record["hit_rate"] = hit_rate(retrieved_ids, relevant_ids, k=TOP_K)
        record["mrr"] = mrr(retrieved_ids, relevant_ids)
        record["precision_at_5"] = precision_at_k(retrieved_ids, relevant_ids, k=TOP_K)
        record["average_precision"] = average_precision(retrieved_ids, relevant_ids, k=TOP_K)
        record["ndcg@5"] = ndcg_at_k(retrieved_ids, relevant_ids, k=TOP_K)
    return record


def _run_adversarial_probes() -> list[dict]:
    probes = []
    for probe in ADVERSARIAL_PROBES:
        graded = {}
        provider = None
        for label, answer in probe["answers"].items():
            judge = evaluate_answer(probe["query"], probe["context"], answer)
            provider = judge.provider
            graded[label] = {
                "answer": answer,
                "faithfulness_1to5": judge.faithfulness_score,
                "relevance_1to5": judge.relevance_score,
                "exact_match": exact_match(answer, probe["reference_answer"]),
                "token_f1": round(token_f1(answer, probe["reference_answer"]), 3),
                "faithfulness_rationale": judge.faithfulness_rationale,
            }
        probes.append(
            {
                "query": probe["query"],
                "judge_provider": provider,
                "graded": graded,
            }
        )
    return probes


def _aggregate(results: list[dict], probes: list[dict]) -> dict:
    answerable = [r for r in results if not r["expected_refusal"] and r["error"] is None]
    refusals = [r for r in results if r["expected_refusal"] and r["error"] is None]
    failed = [{"query": r["query"], "error": r["error"]} for r in results if r["error"]]

    def cold(field):
        vals = [r[field] for r in results if r["error"] is None]
        return {"p50": round(_percentile(vals, 50), 2), "p95": round(_percentile(vals, 95), 2)}

    probe_correct = [p["graded"]["correct"]["faithfulness_1to5"] for p in probes]
    probe_wrong = [p["graded"]["confidently_wrong"]["faithfulness_1to5"] for p in probes]

    is_mock = settings.embedding_provider.lower() == "mock"
    refusal_note = (
        "Distance-gated refusal needs real semantic embeddings; bag-of-words mock embeddings "
        "cannot separate out-of-corpus queries, so refusal is validated by unit tests "
        "(tests/test_generation.py) and the live smoke (reports/smoke_results.json)."
        if is_mock
        else None
    )
    answer_note = (
        "Offline run: system answers come from the mock generator (a placeholder), so per-case "
        "EM/F1/faithfulness reflect that, not real LLM output. Real answer quality is shown by the "
        "adversarial probes and the live smoke."
        if settings.generation_provider.lower() == "mock"
        else None
    )

    return {
        "case_count": len(results),
        "answerable_count": len(answerable),
        "refusal_count": len(refusals),
        "failed_count": len(failed),
        "retrieval": {
            "mean_recall@5": round(_mean(answerable, "recall@5"), 4),
            "mean_hit_rate": round(_mean(answerable, "hit_rate"), 4),
            "mean_mrr": round(_mean(answerable, "mrr"), 4),
            "mean_ndcg@5": round(_mean(answerable, "ndcg@5"), 4),
            "mean_precision_at_5": round(_mean(answerable, "precision_at_5"), 4),
            "mean_average_precision": round(_mean(answerable, "average_precision"), 4),
        },
        "answer_quality": {
            "mean_exact_match": round(_mean(answerable, "exact_match"), 4),
            "mean_token_f1": round(_mean(answerable, "token_f1"), 4),
            "mean_faithfulness_1to5": round(_mean(answerable, "faithfulness_1to5"), 3),
            "mean_relevance_1to5": round(_mean(answerable, "relevance_1to5"), 3),
            "note": answer_note,
        },
        "refusal": {
            "refusal_accuracy": round(_mean(refusals, "refusal_correct"), 4) if refusals else None,
            "refused_count": sum(1 for r in refusals if r.get("refused")),
            "refusal_total": len(refusals),
            "note": refusal_note,
        },
        "adversarial_probe_summary": {
            "mean_faithfulness_correct": (
                round(sum(probe_correct) / len(probe_correct), 3) if probe_correct else None
            ),
            "mean_faithfulness_confidently_wrong": (
                round(sum(probe_wrong) / len(probe_wrong), 3) if probe_wrong else None
            ),
            "judge_discriminates": bool(
                probe_correct and probe_wrong and min(probe_correct) > max(probe_wrong)
            ),
        },
        "latency_cold_ms": {
            "embedding": cold("embedding_latency_ms"),
            "retrieval": cold("retrieval_latency_ms"),
            "generation": cold("generation_latency_ms"),
            "total": cold("total_latency_ms"),
        },
        "total_generation_tokens": sum(r.get("token_usage", 0) for r in results if r["error"] is None),
        "failed_cases": failed,
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": TOP_K,
            "max_retrieval_distance": settings.max_retrieval_distance,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": "mock" if settings.embedding_provider == "mock" else settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "generation_provider": settings.generation_provider,
            "judge_provider": settings.judge_provider,
            "judge_model": settings.judge_model or None,
        },
    }


def _write_report(report: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def run_evaluation(test_set_path="eval/test_set.json", output_path: str | None = None):
    with open(test_set_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        print("Empty test set.")
        return None

    if output_path is None:
        output_path = os.path.join(settings.reports_path, "evaluation_results.json")

    results: list[dict] = []
    probes = _run_adversarial_probes()

    for i, item in enumerate(dataset, start=1):
        try:
            record = _evaluate_case(item)
        except Exception as exc:  # noqa: BLE001 - we record and continue
            record = {
                "query": item.get("query", ""),
                "expected_refusal": item.get("expected_refusal", False),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[case {i}] ERROR: {record['error']}")
            traceback.print_exc()
        results.append(record)

        # Persist partial results after every case so a mid-run failure is never lost.
        report = {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": _aggregate(results, probes),
            "results": results,
            "adversarial_probes": probes,
        }
        _write_report(report, output_path)

    summary = report["summary"]
    print("=== Evaluation Summary ===")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "case_count",
                    "answerable_count",
                    "refusal_count",
                    "failed_count",
                    "retrieval",
                    "answer_quality",
                    "refusal",
                    "adversarial_probe_summary",
                    "latency_cold_ms",
                )
            },
            indent=2,
        )
    )
    if summary["failed_cases"]:
        print(f"\nWARNING: {len(summary['failed_cases'])} case(s) failed:")
        for fc in summary["failed_cases"]:
            print(f"  - {fc['query'][:60]!r}: {fc['error']}")
    print(f"\nSaved detailed results to {output_path}")
    return report


if __name__ == "__main__":
    run_evaluation()
