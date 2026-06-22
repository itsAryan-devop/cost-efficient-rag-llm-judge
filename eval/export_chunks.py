import json
import os
from datetime import UTC, datetime

from src.config import settings
from src.storage import get_table


def run(output_path=None):
    table = get_table()
    rows = table.to_arrow().to_pylist()
    chunks = [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "source_file": row["source_file"],
            "doc_type": row["doc_type"],
            "chunk_index": row["chunk_index"],
            "text_preview": row["text"][:500],
        }
        for row in rows
    ]

    report = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    if output_path is None:
        os.makedirs(settings.reports_path, exist_ok=True)
        output_path = os.path.join(settings.reports_path, "chunk_inventory.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(chunks)} chunks to {output_path}")
    return report


if __name__ == "__main__":
    run()
