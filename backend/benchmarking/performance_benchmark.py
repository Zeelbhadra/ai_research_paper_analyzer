"""Isolated performance measurement: one fresh process per candidate (clean memory baseline, no cache sharing).

    python -m backend.benchmarking.performance_benchmark --candidate bge --out x.json   # one candidate
    python -m backend.benchmarking.performance_benchmark --all                           # driver: subprocess each

Measured (not estimated) unless marked extrapolated:
  load_seconds, memory_mb (RSS growth after load + warm queries), latency p50/p95/mean ms (sequential, single
  client, warm), throughput_qps, embedding speed on a fixed random sample of chunks (seconds/chunk),
  index_bytes (on disk). indexing_seconds for dense models = sec/chunk(sample) x corpus size + index build:
  EXTRAPOLATED from the sample (full 30-minute re-embeds per candidate are impractical on CPU).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

from backend.config.settings import get_settings


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def measure(candidate: str, cfg_key: str, n_sample: int = 192, rounds: int = 3, warmup: int = 5, seed: int = 42) -> dict:
    from backend.benchmarking.dataset_builder import load_dataset
    from backend.database.repositories import Repository
    from backend.embeddings.base import DenseEmbedder, make_embedder
    from backend.embeddings.embedding_manager import EmbeddingManager
    from backend.embeddings.tfidf import TfidfIndex
    from backend.retrieval.factory import build_retriever
    from backend.vectorstore.faiss_store import FaissStore

    s = get_settings()
    mgr = EmbeddingManager(s.storage_dir)
    chunks = Repository(s.db_path).get_chunks(cfg_key)
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    queries = [r["query"] for r in ds["records"] if not r.get("excluded_reason")]
    sample = random.Random(seed).sample(chunks, min(n_sample, len(chunks)))
    base = _rss_mb()

    t0 = time.perf_counter()
    r = build_retriever(candidate, mgr, cfg_key)
    load_s = time.perf_counter() - t0
    health = bool(r.health())
    for q in queries[:warmup]:
        r.retrieve(q, 10)
    lat = []
    t_all = time.perf_counter()
    for _ in range(rounds):
        for q in queries:
            t = time.perf_counter()
            r.retrieve(q, 10)
            lat.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t_all
    mem = _rss_mb() - base
    lat_sorted = sorted(lat)

    # --- indexing cost ---
    parts = [candidate] if not candidate.startswith("hybrid:") else ["tfidf", candidate.split(":")[1]]
    index_s, index_bytes, sec_per_chunk, extrap = 0.0, 0, {}, False
    for p in parts:
        if p == "tfidf":
            t = time.perf_counter()
            idx = TfidfIndex().fit([c.chunk_id for c in chunks], [c.text for c in chunks])   # full corpus, real
            index_s += time.perf_counter() - t
            index_bytes += sum(f.stat().st_size for f in (next((mgr.root / "indexes").glob("tfidf_*/" + cfg_key))).glob("*"))
        else:
            e: DenseEmbedder = make_embedder(p)
            e.load()
            texts = [c.text for c in sample]
            e.embed_documents(texts[:8], 32)                                   # warm-up
            t = time.perf_counter()
            X = e.embed_documents(texts, 32)
            dt = time.perf_counter() - t
            spc = dt / len(texts)
            sec_per_chunk[p] = spc
            t = time.perf_counter()
            st = FaissStore(e.dim, e.version)
            st.add([c.chunk_id for c in sample], X)
            build = (time.perf_counter() - t) * len(chunks) / len(sample)
            index_s += spc * len(chunks) + build
            index_bytes += sum(f.stat().st_size for f in mgr.index_dir(e.key, cfg_key).glob("*"))
            extrap = True
    return {"candidate": candidate, "n_chunks": len(chunks), "n_queries": len(queries), "rounds": rounds,
            "health_ok": health, "load_seconds": round(load_s, 3), "memory_mb": round(mem, 1),
            "latency_mean_ms": round(statistics.mean(lat), 3), "latency_p50_ms": round(lat_sorted[len(lat) // 2], 3),
            "latency_p95_ms": round(lat_sorted[int(len(lat) * 0.95) - 1], 3), "throughput_qps": round(len(lat) / wall, 3),
            "embed_seconds_per_chunk": {k: round(v, 5) for k, v in sec_per_chunk.items()},
            "indexing_seconds": round(index_s, 2), "indexing_extrapolated": extrap, "index_bytes": index_bytes,
            "sample_size": len(sample), "device": "cpu", "cpu_threads": psutil.cpu_count(logical=True)}


def run_all(candidates: list[str], cfg_key: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for c in candidates:
        f = out_dir / f"perf_{c.replace(':', '_')}.json"
        print(f"[perf] {c} (fresh process)", flush=True)
        subprocess.run([sys.executable, "-m", "backend.benchmarking.performance_benchmark", "--candidate", c,
                        "--chunk-cfg", cfg_key, "--out", str(f)], check=True, stdout=subprocess.DEVNULL)
        results[c] = json.loads(f.read_text())
        print("   ", {k: results[c][k] for k in ("latency_p50_ms", "latency_p95_ms", "throughput_qps", "memory_mb", "indexing_seconds")}, flush=True)
    (out_dir / "performance.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate")
    ap.add_argument("--all", nargs="*", help="candidates to run, each in a fresh subprocess")
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.candidate:
        res = measure(a.candidate, a.chunk_cfg)
        Path(a.out).write_text(json.dumps(res, indent=1)) if a.out else print(json.dumps(res, indent=1))
    else:
        run_all(a.all or ["tfidf", "sbert", "e5", "bge", "hybrid:e5:1.0:0.5"], a.chunk_cfg, get_settings().storage_dir / "benchmark")
