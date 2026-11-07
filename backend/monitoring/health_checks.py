"""Retriever health probing: component health + a real smoke query, with latency."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class HealthReport:
    ok: bool
    latency_ms: float
    detail: str


def probe(retriever, smoke_query: str = "neural network") -> HealthReport:
    t = time.perf_counter()
    try:
        if not retriever.health():
            return HealthReport(False, (time.perf_counter() - t) * 1000, "component health check failed")
        hits = retriever.retrieve(smoke_query, 3)
        if not hits:
            return HealthReport(False, (time.perf_counter() - t) * 1000, "smoke query returned no hits")
        return HealthReport(True, (time.perf_counter() - t) * 1000, "ok")
    except Exception as e:  # any failure is a failed probe, never a crash
        return HealthReport(False, (time.perf_counter() - t) * 1000, f"{type(e).__name__}: {e}")
