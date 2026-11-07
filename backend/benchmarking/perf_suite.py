"""Measured performance checks (not estimates).   python -m backend.benchmarking.perf_suite <concurrency|large_doc|batch_embed|scale|all>

  concurrency  real production retriever (TF-IDF + E5) under N parallel clients
  large_doc    extraction + cleaning + sectioning + chunking of a generated 300-page PDF (time and peak memory)
  batch_embed  SBERT batch-size sweep on real chunks (chunks/s)
  scale        TF-IDF retrieval over a synthetic corpus of 20,000 chunks (index time, latency, memory, size)
Results are written to storage/benchmark/perf_suite.json (merged per section)."""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil

from backend.config.settings import get_settings


def _save(section: str, data: dict) -> None:
    p = get_settings().storage_dir / "benchmark" / "perf_suite.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[section] = data
    p.write_text(json.dumps(cur, indent=1))


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def concurrency(workers_list=(1, 2, 4, 8), n_queries: int = 48) -> dict:
    from backend.benchmarking.dataset_builder import load_dataset
    from backend.retrieval.production import ProductionRetrieval
    s = get_settings()
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    queries = [r["query"] for r in ds["records"] if not r.get("excluded_reason")]
    queries = (queries * 2)[:n_queries]
    pr = ProductionRetrieval()
    for q in queries[:3]:
        pr.search(q, 10)                                                    # warm-up
    out = {"production": pr.service.status.primary, "n_queries": n_queries, "cpu_threads": psutil.cpu_count(logical=True), "runs": {}}
    for w in workers_list:
        lat, errs = [], 0

        def one(q):
            nonlocal errs
            t = time.perf_counter()
            try:
                r = pr.search(q, 10)
                assert r["results"]
            except Exception:
                errs += 1
            lat.append((time.perf_counter() - t) * 1000)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(one, queries))
        wall = time.perf_counter() - t0
        out["runs"][str(w)] = {"qps": round(len(queries) / wall, 2), "p50_ms": round(_pct(lat, 50), 1), "p95_ms": round(_pct(lat, 95), 1), "errors": errs}
        print("workers", w, out["runs"][str(w)], flush=True)
    _save("concurrency", out)
    return out


def large_doc(pages: int = 300) -> dict:
    import fitz
    from backend.ingestion.pdf_extractor import extract_pdf
    from backend.preprocessing.chunking import ChunkConfig, chunk_document
    from backend.preprocessing.section_extractor import build_structured_text
    from backend.preprocessing.text_cleaner import clean_pages
    rng = random.Random(42)
    vocab = [f"term{i}" for i in range(800)]
    sections = ["Abstract", "1 Introduction", "2 Related Work", "3 Methodology", "4 Results", "5 Conclusion", "References"]
    doc = fitz.open()
    for i in range(pages):
        pg = doc.new_page()
        pg.insert_text((72, 40), "Journal of Synthetic Load Tests", fontsize=9)
        y = 80
        if i % (pages // len(sections) + 1) == 0:
            pg.insert_text((72, y), sections[min(i // (pages // len(sections) + 1), len(sections) - 1)], fontsize=14)
            y += 28
        body = " ".join(rng.choice(vocab) for _ in range(330)) + "."
        pg.insert_textbox(fitz.Rect(72, y, 520, 780), body, fontsize=10)
        pg.insert_text((290, 810), str(i + 1), fontsize=9)
    data = doc.tobytes()
    doc.close()
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    tracemalloc.start()
    t = time.perf_counter()
    ex = extract_pdf(data=data)
    t_ex = time.perf_counter() - t
    t = time.perf_counter()
    pages_clean = clean_pages(ex.pages)
    st = build_structured_text(pages_clean)
    t_clean = time.perf_counter() - t
    t = time.perf_counter()
    chunks = chunk_document(st, "big", "Synthetic", "u", ChunkConfig("recursive", 256, 0.15))
    t_chunk = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out = {"pages": pages, "pdf_mb": round(len(data) / 1e6, 2), "chars": ex.n_chars, "chunks": len(chunks), "extract_s": round(t_ex, 2), "clean_sections_s": round(t_clean, 2),
           "chunk_s": round(t_chunk, 2), "total_s": round(t_ex + t_clean + t_chunk, 2), "pages_per_s": round(pages / (t_ex + t_clean + t_chunk), 1),
           "python_heap_peak_mb": round(peak / 2**20, 1), "rss_growth_mb": round((proc.memory_info().rss - rss0) / 2**20, 1),
           "sections_detected": sorted(set(st.section_names))}
    print(out, flush=True)
    _save("large_doc", out)
    return out


def batch_embed(sizes=(8, 32, 64), n: int = 192) -> dict:
    from backend.database.repositories import Repository
    from backend.embeddings.base import make_embedder
    s = get_settings()
    chunks = Repository(s.db_path).get_chunks("recursive-256-15")
    texts = [c.text for c in random.Random(42).sample(chunks, n)]
    e = make_embedder("sbert")
    e.load()
    e.embed_documents(texts[:8], 8)
    out = {"model": e.checkpoint, "device": e.device, "n_chunks": n, "batch_sizes": {}}
    for b in sizes:
        t = time.perf_counter()
        X = e.embed_documents(texts, b)
        dt = time.perf_counter() - t
        out["batch_sizes"][str(b)] = {"chunks_per_s": round(n / dt, 1), "seconds": round(dt, 2)}
        assert X.shape == (n, e.dim) and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
        print("batch", b, out["batch_sizes"][str(b)], flush=True)
    _save("batch_embed", out)
    return out


def scale(n_chunks: int = 20000, n_queries: int = 200) -> dict:
    from backend.embeddings.tfidf import TfidfIndex
    rng = random.Random(42)
    vocab = [f"w{i}x" for i in range(5000)]
    texts = [" ".join(rng.choice(vocab) for _ in range(180)) for _ in range(n_chunks)]
    ids = [f"c{i}" for i in range(n_chunks)]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    idx = TfidfIndex().fit(ids, texts)
    t_fit = time.perf_counter() - t
    qs = [" ".join(rng.choice(vocab) for _ in range(8)) for _ in range(n_queries)]
    lat = []
    for q in qs:
        t = time.perf_counter()
        idx.search(q, 10)
        lat.append((time.perf_counter() - t) * 1000)
    tmp = get_settings().storage_dir / "perf_tmp_tfidf"
    idx.save(tmp)
    size = sum(f.stat().st_size for f in tmp.glob("*"))
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    out = {"chunks": n_chunks, "index_s": round(t_fit, 2), "p50_ms": round(_pct(lat, 50), 1), "p95_ms": round(_pct(lat, 95), 1), "qps_single_thread": round(1000 / statistics.mean(lat), 1),
           "memory_growth_mb": round((proc.memory_info().rss - rss0) / 2**20, 1), "index_mb_on_disk": round(size / 2**20, 1)}
    print(out, flush=True)
    _save("scale_tfidf", out)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"concurrency": concurrency, "large_doc": large_doc, "batch_embed": batch_embed, "scale": scale}
    for name, fn in fns.items():
        if which in (name, "all"):
            fn()
