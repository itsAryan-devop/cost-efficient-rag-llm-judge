import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.config import settings


@dataclass
class CostAssumptions:
    vector_dimension: int = 768
    bytes_per_float: int = 4
    metadata_overhead_bytes_per_vector: int = 700
    storage_cost_per_gb_month: float = 0.08
    existing_app_server_cost_month: float = 10.0
    managed_base_cost_month: float = 70.0
    managed_cost_per_extra_million_vectors: float = 10.0
    embedding_cost_per_million_input_tokens: float = 0.15
    avg_tokens_per_chunk: int = 180


def estimate_index_gb(vector_count: int, assumptions: CostAssumptions) -> float:
    bytes_per_vector = (
        assumptions.vector_dimension * assumptions.bytes_per_float
        + assumptions.metadata_overhead_bytes_per_vector
    )
    return vector_count * bytes_per_vector / (1024 ** 3)


def estimate_rows(scales=(100_000, 1_000_000, 10_000_000), assumptions=None):
    assumptions = assumptions or CostAssumptions(vector_dimension=settings.embedding_dimension)
    rows = []
    for vector_count in scales:
        index_gb = estimate_index_gb(vector_count, assumptions)
        lancedb_storage_cost = index_gb * assumptions.storage_cost_per_gb_month
        extra_millions = max(vector_count - 1_000_000, 0) / 1_000_000
        managed_cost = (
            assumptions.managed_base_cost_month
            + extra_millions * assumptions.managed_cost_per_extra_million_vectors
        )
        embedding_tokens = vector_count * assumptions.avg_tokens_per_chunk
        embedding_cost = (
            embedding_tokens / 1_000_000
            * assumptions.embedding_cost_per_million_input_tokens
        )
        rows.append({
            "vectors": vector_count,
            "estimated_index_gb": round(index_gb, 3),
            "lancedb_storage_cost_month": round(lancedb_storage_cost, 2),
            "managed_vector_db_cost_month": round(managed_cost, 2),
            "one_time_embedding_cost": round(embedding_cost, 2),
        })
    return assumptions, rows


def to_markdown(rows):
    lines = [
        "| Vectors | Est. Index Size (GB) | LanceDB Storage ($/mo) | Managed DB Est. ($/mo) | One-time Embedding Cost ($) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['vectors']:,} | {row['estimated_index_gb']} | "
            f"{row['lancedb_storage_cost_month']} | {row['managed_vector_db_cost_month']} | "
            f"{row['one_time_embedding_cost']} |"
        )
    return "\n".join(lines)


def run(output_path=None):
    assumptions, rows = estimate_rows()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assumptions": asdict(assumptions),
        "rows": rows,
        "markdown_table": to_markdown(rows),
    }

    if output_path is None:
        os.makedirs(settings.reports_path, exist_ok=True)
        output_path = os.path.join(settings.reports_path, "cost_analysis.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(report["markdown_table"])
    print(f"Saved cost analysis to {output_path}")
    return report


if __name__ == "__main__":
    run()
