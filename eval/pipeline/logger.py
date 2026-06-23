import json
import os
import threading
from datetime import datetime, timezone  # noqa: F401  (datetime kept for downstream subclasses)
from .schemas import AuditLogEntry
from .config import pipeline_settings


class AuditLogger:
    """Thread-safe JSONL audit logger for judge calls."""

    def __init__(self, path: str | None = None):
        self.path = path or pipeline_settings.audit_log_path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._entries: list[AuditLogEntry] = []

    def log(self, entry: AuditLogEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")

    @property
    def entries(self) -> list[AuditLogEntry]:
        return list(self._entries)

    def total_tokens(self) -> tuple[int, int, int]:
        prompt = sum(e.prompt_tokens for e in self._entries)
        completion = sum(e.completion_tokens for e in self._entries)
        return prompt, completion, prompt + completion

    def total_cost(self) -> float:
        return sum(e.cost_estimate for e in self._entries)

    def total_calls(self) -> int:
        return len(self._entries)
