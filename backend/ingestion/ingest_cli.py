"""Ingest a directory of PDFs: extract, clean, section, chunk and store.
    python -m backend.ingestion.ingest_cli --dir data [--chunk-cfg recursive-256-15]
Then build indexes with `python -m backend.benchmarking.embed_corpus` (see DEPLOYMENT.md)."""
from __future__ import annotations

import argparse
import logging

from backend.agents.document_agent import parse_cfg_key
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.ingestion.document_processor import ingest_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="directory of PDFs (default: DATA_DIR)")
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    a = ap.parse_args()
    logging.basicConfig(level="WARNING")
    s = get_settings()
    stats = ingest_directory(Repository(s.db_path), a.dir or s.data_dir, parse_cfg_key(a.chunk_cfg))
    print(stats)


if __name__ == "__main__":
    main()
