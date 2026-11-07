"""Persistent model registry: every benchmark run is kept (history), one production pointer.

JSON is the source of truth (storage/registry/registry.json); a flat CSV is regenerated for inspection.
Writes are atomic (temp file + replace).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "registry.json"
        self._lock = threading.RLock()
        self.data: dict[str, Any] = {"runs": [], "production": None, "promotions": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def _save(self) -> None:
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, self.path)
            self._write_csv()

    def _write_csv(self) -> None:
        keys = ["run_id", "timestamp", "dataset_version", "chunk_cfg", "selection_split", "name", "family", "checkpoint",
                "version", "dim", "overall_score", "eligible", "is_winner", "is_production", "index_location",
                "nDCG@10", "Recall@5", "MRR@10", "Precision@5", "MAP", "latency_p50_ms", "latency_p95_ms",
                "throughput_qps", "memory_mb", "indexing_seconds", "index_bytes"]
        prod = self.data.get("production") or {}
        with open(self.dir / "registry.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in self.data["runs"]:
                for c in r["candidates"]:
                    row = {**r["meta"], "run_id": r["run_id"], "timestamp": r["timestamp"], **c, **c.get("metrics", {}),
                           "is_winner": c["name"] == (r.get("winner") or {}).get("name"),
                           "is_production": c["version"] == prod.get("version") and r["run_id"] == prod.get("run_id")}
                    w.writerow([row.get(k) for k in keys])

    # ---- runs ----
    def record_run(self, meta: dict, candidates: list[dict], winner: dict | None, notes: list[str] | None = None) -> str:
        """meta: dataset_version, chunk_cfg, selection_split, weights, thresholds, seed ...
        candidate: name, family, checkpoint, version, dim, metrics{...}, overall_score, eligible, fail_reasons,
        index_location, config{...}."""
        with self._lock:
            run_id = f"run{len(self.data['runs']) + 1:04d}"
            self.data["runs"].append({"run_id": run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "meta": meta, "candidates": candidates, "winner": winner, "notes": notes or []})
            self._save()
            return run_id

    def get_run(self, run_id: str) -> dict:
        for r in self.data["runs"]:
            if r["run_id"] == run_id:
                return r
        raise KeyError(run_id)

    def latest_run(self) -> dict | None:
        return self.data["runs"][-1] if self.data["runs"] else None

    def attach(self, run_id: str, key: str, value: Any) -> None:
        """Attach extra results (e.g. held-out test metrics) to a recorded run."""
        with self._lock:
            self.get_run(run_id).setdefault("attachments", {})[key] = value
            self._save()

    # ---- production ----
    @property
    def production(self) -> dict | None:
        return self.data.get("production")

    def promote(self, run_id: str, candidate_name: str, reason: str) -> dict:
        with self._lock:
            run = self.get_run(run_id)
            cand = next(c for c in run["candidates"] if c["name"] == candidate_name)
            if not cand.get("eligible", False):
                raise ValueError(f"{candidate_name} did not pass thresholds: {cand.get('fail_reasons')}")
            prev = self.data.get("production")
            new = {"name": cand["name"], "family": cand["family"], "version": cand["version"], "checkpoint": cand["checkpoint"],
                   "index_location": cand["index_location"], "run_id": run_id, "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "overall_score": cand["overall_score"], "config": cand.get("config", {})}
            self.data["promotions"].append({"at": new["promoted_at"], "from": prev and prev["name"], "to": new["name"],
                                            "run_id": run_id, "reason": reason})
            self.data["production"] = new
            self._save()
            return new

    def history(self) -> list[dict]:
        return list(self.data["promotions"])
