"""Reproducible, apples-to-apples vector-store cost comparison.

Embedded LanceDB still needs an always-on host to serve queries, so its total is
**storage + compute/host**, which is what makes it comparable to a managed
service that bundles compute and storage. The managed baseline is a named,
publicly-priced option (Pinecone serverless, Standard plan) rather than a
hand-wavy number, with the pricing date and source recorded below.

All figures are conservative monthly estimates; read/write traffic costs on the
managed side scale with query volume and are called out separately.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

from src.config import settings

# Managed baseline pricing, recorded with its source and date so the estimate is
# auditable. Source: https://www.pinecone.io/pricing/ and
# https://docs.pinecone.io/guides/manage-cost/understanding-cost
MANAGED_PROVIDER = "Pinecone serverless (Standard plan)"
MANAGED_PRICING_SOURCE = "https://www.pinecone.io/pricing/"
MANAGED_PRICING_DATE = "2026-06-23"


@dataclass
class CostAssumptions:
    vector_dimension: int = 768
    bytes_per_float: int = 4
    metadata_overhead_bytes_per_vector: int = 700

    # LanceDB: cheap object/local storage PLUS an always-on host to serve it.
    lancedb_storage_cost_per_gb_month: float = 0.08
    lancedb_host_cost_month: float = 10.0  # configurable via LANCEDB_HOST_COST_MONTH

    # Managed (Pinecone serverless Standard): storage per GB-month, a monthly plan
    # minimum you pay even when usage is lower, plus per-operation read/write units.
    managed_storage_cost_per_gb_month: float = 0.33
    managed_plan_minimum_month: float = 50.0
    managed_read_cost_per_million_ru: float = 8.25
    managed_write_cost_per_million_wu: float = 2.00

    # One-time embedding (indexing) cost.
    embedding_cost_per_million_input_tokens: float = 0.15
    # CHUNK_SIZE / CHUNK_OVERLAP are CHARACTERS (RecursiveCharacterTextSplitter).
    # Embedding bills tokens, so convert: ~4.4 chars/token for English text.
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 200
    chars_per_token: float = 4.4

    @property
    def avg_tokens_per_chunk(self) -> float:
        """Derived, not assumed: effective (non-overlapping) chars per chunk / chars-per-token."""
        effective_chars = max(self.chunk_size_chars - self.chunk_overlap_chars, 1)
        return effective_chars / self.chars_per_token


def estimate_index_gb(vector_count: int, assumptions: CostAssumptions) -> float:
    bytes_per_vector = (
        assumptions.vector_dimension * assumptions.bytes_per_float
        + assumptions.metadata_overhead_bytes_per_vector
    )
    return vector_count * bytes_per_vector / (1024**3)


def estimate_rows(scales=(100_000, 1_000_000, 10_000_000), assumptions=None):
    if assumptions is None:
        assumptions = CostAssumptions(
            vector_dimension=settings.embedding_dimension,
            lancedb_host_cost_month=float(
                os.environ.get("LANCEDB_HOST_COST_MONTH", CostAssumptions.lancedb_host_cost_month)
            ),
            chunk_size_chars=settings.chunk_size,
            chunk_overlap_chars=settings.chunk_overlap,
        )

    rows = []
    for vector_count in scales:
        index_gb = estimate_index_gb(vector_count, assumptions)

        lancedb_storage = index_gb * assumptions.lancedb_storage_cost_per_gb_month
        lancedb_total = assumptions.lancedb_host_cost_month + lancedb_storage

        managed_storage = index_gb * assumptions.managed_storage_cost_per_gb_month
        # You pay at least the plan minimum; light query traffic is covered by it.
        managed_total = max(assumptions.managed_plan_minimum_month, managed_storage)

        embedding_tokens = vector_count * assumptions.avg_tokens_per_chunk
        embedding_cost = embedding_tokens / 1_000_000 * assumptions.embedding_cost_per_million_input_tokens

        rows.append(
            {
                "vectors": vector_count,
                "estimated_index_gb": round(index_gb, 3),
                "lancedb_storage_cost_month": round(lancedb_storage, 2),
                "lancedb_host_cost_month": round(assumptions.lancedb_host_cost_month, 2),
                "lancedb_total_cost_month": round(lancedb_total, 2),
                "managed_storage_cost_month": round(managed_storage, 2),
                "managed_total_cost_month": round(managed_total, 2),
                "one_time_embedding_cost": round(embedding_cost, 2),
            }
        )
    return assumptions, rows


def to_markdown(rows):
    lines = [
        "| Vectors | Index Size (GB) | LanceDB storage ($/mo) | LanceDB host ($/mo) | "
        "**LanceDB total ($/mo)** | **Managed total ($/mo)** | One-time embedding ($) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['vectors']:,} | {row['estimated_index_gb']} | "
            f"{row['lancedb_storage_cost_month']} | {row['lancedb_host_cost_month']} | "
            f"**{row['lancedb_total_cost_month']}** | **{row['managed_total_cost_month']}** | "
            f"{row['one_time_embedding_cost']} |"
        )
    return "\n".join(lines)


def break_even_note(assumptions: CostAssumptions) -> str:
    return (
        f"At all three scales, embedded LanceDB (~${assumptions.lancedb_host_cost_month:.0f}/mo host "
        f"+ cents of storage) is roughly "
        f"{assumptions.managed_plan_minimum_month / assumptions.lancedb_host_cost_month:.0f}x cheaper "
        f"than {MANAGED_PROVIDER} (>=${assumptions.managed_plan_minimum_month:.0f}/mo plan minimum "
        f"+ ${assumptions.managed_storage_cost_per_gb_month}/GB storage + read/write units). "
        "Managed wins once you need multi-region HA, managed backups/SLAs, or such high query "
        "concurrency that the always-on host you must run for LanceDB would itself have to be larger "
        f"than the ${assumptions.managed_plan_minimum_month:.0f}/mo managed plan -- i.e. the cost "
        "break-even is the point where your required compute exceeds the managed minimum, after "
        "which serverless autoscaling can be cheaper than an over-provisioned host."
    )


def run(output_path=None):
    assumptions, rows = estimate_rows()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_date": date.today().isoformat(),
        "managed_baseline": {
            "provider": MANAGED_PROVIDER,
            "source": MANAGED_PRICING_SOURCE,
            "pricing_date": MANAGED_PRICING_DATE,
            "storage_cost_per_gb_month": assumptions.managed_storage_cost_per_gb_month,
            "plan_minimum_month": assumptions.managed_plan_minimum_month,
            "read_cost_per_million_ru": assumptions.managed_read_cost_per_million_ru,
            "write_cost_per_million_wu": assumptions.managed_write_cost_per_million_wu,
            "note": "Read/write unit costs scale with query volume and are additional to the totals "
            "above; for a lightly-queried index they are covered by the plan minimum.",
        },
        "assumptions": {
            **asdict(assumptions),
            "avg_tokens_per_chunk_derived": round(assumptions.avg_tokens_per_chunk, 1),
            "chunk_size_unit": "characters (RecursiveCharacterTextSplitter)",
        },
        "rows": rows,
        "break_even": break_even_note(assumptions),
        "markdown_table": to_markdown(rows),
    }

    if output_path is None:
        os.makedirs(settings.reports_path, exist_ok=True)
        output_path = os.path.join(settings.reports_path, "cost_analysis.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(report["markdown_table"])
    print("\n" + report["break_even"])
    print(f"\nSaved cost analysis to {output_path}")
    return report


if __name__ == "__main__":
    run()
