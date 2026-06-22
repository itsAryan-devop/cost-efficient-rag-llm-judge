import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def log_query(query: str, latency_ms: float, chunk_count: int, token_usage: int = 0, **extra: Any):
    """Logs structured telemetry for a query."""
    log_event(
        "query",
        query=query,
        latency_ms=round(latency_ms, 2),
        chunk_count=chunk_count,
        token_usage=token_usage,
        **extra,
    )
