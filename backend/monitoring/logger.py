"""Structured (JSON-lines) logging and a durable event log with an in-memory tail for dashboards."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def setup_logging(level: str = "INFO", json_logs: bool = False, logfile: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in [logging.StreamHandler()] + ([logging.FileHandler(logfile)] if logfile else []):
        h.setFormatter(fmt)
        root.addHandler(h)


class EventLog:
    """Append-only JSONL file + bounded in-memory tail (thread-safe)."""

    def __init__(self, path: Path | None = None, keep: int = 500):
        self.path, self._tail, self._lock = Path(path) if path else None, deque(maxlen=keep), threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **fields) -> dict:
        ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **fields}
        with self._lock:
            self._tail.append(ev)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def tail(self, n: int = 50, kind: str | None = None) -> list[dict]:
        with self._lock:
            evs = [e for e in self._tail if kind is None or e["kind"] == kind]
        return evs[-n:]
