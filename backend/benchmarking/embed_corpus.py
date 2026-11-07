"""Embed the stored chunks with every configured model and build indexes; writes timings.

    python -m backend.benchmarking.embed_corpus --chunk-cfg recursive-256-15 [--families sbert e5 bge]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.base import make_embedder
from backend.embeddings.embedding_manager import EmbeddingManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    ap.add_argument("--families", nargs="+", default=["sbert", "e5", "bge"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    s = get_settings()
    repo = Repository(s.db_path)
    chunks = repo.get_chunks(a.chunk_cfg)
    print(f"{len(chunks)} chunks, {len({c.document_id for c in chunks})} documents", flush=True)
    mgr = EmbeddingManager(s.storage_dir, a.batch_size)
    out = {}
    t0 = time.perf_counter()
    tf, st = mgr.build_tfidf_index(chunks, a.chunk_cfg)
    out["tfidf"] = {"version": tf.version, "index_seconds": st.index_seconds, "index_bytes": st.index_bytes}
    for fam in a.families:
        e = make_embedder(fam, device=a.device)
        e.load()
        t = time.perf_counter()
        _, st = mgr.build_dense_index(e, chunks, a.chunk_cfg)
        out[fam] = {"checkpoint": e.checkpoint, "version": e.version, "dim": e.dim, "device": e.device,
                    "n_embedded": st.n_embedded, "embed_seconds": round(st.embed_seconds, 2),
                    "index_seconds": round(st.index_seconds, 4), "index_bytes": st.index_bytes,
                    "reused_index": st.reused_index, "wall_seconds": round(time.perf_counter() - t, 2)}
        print(fam, json.dumps(out[fam]), flush=True)
    (s.storage_dir / f"embed_timings_{a.chunk_cfg}.json").write_text(json.dumps(out, indent=2))
    print("total", round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
