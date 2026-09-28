"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request key -> (monotonic start, input text, iso timestamp)
        self._open: dict[str, tuple[float, str, str]] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        self._open[self._key(user_id, request_id)] = (
            time.perf_counter(),
            text,
            utc_now_iso(),
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        started, input_text, input_ts = self._open.pop(
            self._key(user_id, request_id),
            (time.perf_counter(), "", utc_now_iso()),
        )
        self.logs.append(
            {
                "id": request_id or uuid.uuid4().hex[:12],
                "user_id": user_id,
                "input": input_text,
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "input_timestamp": input_ts,
                "output_timestamp": utc_now_iso(),
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
