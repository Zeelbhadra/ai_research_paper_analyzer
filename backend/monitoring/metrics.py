"""Lightweight operational metrics: append-only SQLite table (name, value, tags, ts) with windowed queries.

Recorded online by the serving path (retrieval latency, fallback use, failures, RAG latency/verification) and read by the
monitoring agent. No external metrics stack: one table, no extra service.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


class MetricsStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._c = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._c.executescript("CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, name TEXT, value REAL, tags TEXT);"
                                  "CREATE INDEX IF NOT EXISTS ix_metrics ON metrics(name, ts);")

    def record(self, name: str, value: float, ts: float | None = None, **tags) -> None:
        with self._lock:
            self._c.execute("INSERT INTO metrics(ts,name,value,tags) VALUES (?,?,?,?)", (ts or time.time(), name, float(value), json.dumps(tags)))
            self._c.commit()

    def window(self, name: str, since: float, until: float | None = None) -> list[tuple[float, float, dict]]:
        q, a = "SELECT ts,value,tags FROM metrics WHERE name=? AND ts>=?", [name, since]
        if until is not None:
            q, a = q + " AND ts<?", a + [until]
        with self._lock:
            return [(r[0], r[1], json.loads(r[2])) for r in self._c.execute(q + " ORDER BY ts", a)]

    def stats(self, name: str, since: float, until: float | None = None) -> dict:
        v = np.array([x[1] for x in self.window(name, since, until)])
        if len(v) == 0:
            return {"n": 0}
        return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max()), "sum": float(v.sum())}

    def rate(self, name: str, since: float, until: float | None = None) -> float | None:
        """Mean of a 0/1 metric (e.g. fallback_used) in the window, or None if no data."""
        s = self.stats(name, since, until)
        return None if s["n"] == 0 else s["mean"]
