"""Build the benchmark dataset with a local LLM.   python -m backend.benchmarking.build_dataset"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.benchmarking.dataset_builder import (BuildConfig, assert_no_leakage, build_dataset,
                                                  export_review_csv, save_dataset)
from backend.benchmarking.llm_local import LocalLLM
from backend.config.settings import get_settings
from backend.database.repositories import Repository


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-category", nargs=3, type=int, default=[8, 4, 5], metavar=("DEV", "VAL", "TEST"))
    ap.add_argument("--resume", action="store_true", help="continue from storage/benchmark/partial_records.json")
    a = ap.parse_args()
    s = get_settings()
    chunks = Repository(s.db_path).get_chunks(a.chunk_cfg)
    llm = LocalLLM(a.llm).load()
    t0 = time.perf_counter()
    ds = build_dataset(chunks, lambda m: llm.generate(m, 60), llm.yes_probability,
                       BuildConfig(seed=s.seed, canonical_chunk_cfg=a.chunk_cfg,
                                  per_category=dict(zip(('dev', 'validation', 'test'), a.per_category))),
                       resume=(__import__('json').loads((s.storage_dir / 'benchmark' / 'partial_records.json').read_text()) if a.resume else None),
                       on_progress=lambda recs: Path(s.storage_dir / 'benchmark').mkdir(parents=True, exist_ok=True) or
                       (s.storage_dir / 'benchmark' / 'partial_records.json').write_text(__import__('json').dumps(recs)),
                       log=lambda m: print(f"{time.perf_counter() - t0:7.0f}s {m}", flush=True))
    ds["config"]["generator_llm"] = a.llm
    assert_no_leakage(ds)
    out = Path(a.out or s.storage_dir / "benchmark" / "dataset.json")
    save_dataset(ds, out)
    export_review_csv(ds, out.with_name("review.csv"))
    print("saved", out, "n =", len(ds["records"]), "version", ds["version"], "rejected", ds["rejected"], flush=True)


if __name__ == "__main__":
    main()
