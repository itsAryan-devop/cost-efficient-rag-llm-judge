import json
import os
from datetime import datetime, timezone
from src.api import query_rag, QueryRequest
from src.config import settings
from eval.ir_metrics import recall_at_k, hit_rate, mrr, context_precision, ndcg_at_k
from eval.llm_judge import evaluate_faithfulness, evaluate_relevance

def _mean(rows, field):
    return sum(r[field] for r in rows) / len(rows) if rows else 0.0

def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]

def run_evaluation(test_set_path="eval/test_set.json", output_path: str | None = None):
    with open(test_set_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    
    if not dataset:
        print("Empty test set.")
        return
    
    for item in dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        
        # 1. Run Query
        response = query_rag(QueryRequest(query=query, top_k=5))
        retrieved_ids = [s["id"] for s in response["sources"]]
        answer = response["answer"]
        context_str = "\n".join([s["text"] for s in response["sources"]])
        
        # 2. Compute IR Metrics
        recall = recall_at_k(retrieved_ids, relevant_ids, k=5)
        hr = hit_rate(retrieved_ids, relevant_ids)
        r_mrr = mrr(retrieved_ids, relevant_ids)
        cp = context_precision(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=5)
        
        # 3. Compute LLM Answer Metrics
        faith = evaluate_faithfulness(context_str, answer)
        rel = evaluate_relevance(query, answer)
        
        results.append({
            "query": query,
            "reference_answer": item.get("reference_answer", ""),
            "retrieved_ids": retrieved_ids,
            "relevant_chunk_ids": relevant_ids,
            "answer": answer,
            "recall@5": recall,
            "hit_rate": hr,
            "mrr": r_mrr,
            "context_precision": cp,
            "ndcg@5": ndcg,
            "faithfulness": faith,
            "relevance": rel,
            "latency_ms": response.get("latency_ms", 0),
            "embedding_latency_ms": response.get("embedding_latency_ms", 0),
            "retrieval_latency_ms": response.get("retrieval_latency_ms", 0),
            "generation_latency_ms": response.get("generation_latency_ms", 0),
            "token_usage": response.get("token_usage", 0),
        })
        
    # Aggregate
    agg = {
        "case_count": len(results),
        "mean_recall@5": _mean(results, "recall@5"),
        "mean_hit_rate": _mean(results, "hit_rate"),
        "mean_mrr": _mean(results, "mrr"),
        "mean_context_precision": _mean(results, "context_precision"),
        "mean_ndcg@5": _mean(results, "ndcg@5"),
        "mean_faithfulness": _mean(results, "faithfulness"),
        "mean_relevance": _mean(results, "relevance"),
        "total_token_usage": sum(r["token_usage"] for r in results),
        "p50_latency_ms": _percentile([r["latency_ms"] for r in results], 50),
        "p95_latency_ms": _percentile([r["latency_ms"] for r in results], 95),
        "p50_retrieval_latency_ms": _percentile([r["retrieval_latency_ms"] for r in results], 50),
        "p95_retrieval_latency_ms": _percentile([r["retrieval_latency_ms"] for r in results], 95),
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "min_relevance_score": settings.min_relevance_score,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "generation_provider": settings.generation_provider,
            "judge_provider": settings.judge_provider,
        },
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "summary": agg,
        "results": results,
    }

    if output_path is None:
        os.makedirs(settings.reports_path, exist_ok=True)
        output_path = os.path.join(settings.reports_path, "evaluation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("=== Evaluation Results ===")
    print(json.dumps(agg, indent=2))
    print(f"Saved detailed results to {output_path}")
    return report
    
if __name__ == "__main__":
    run_evaluation()
