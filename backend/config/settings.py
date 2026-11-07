"""Central configuration. Everything tunable lives here or in env vars (no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data")))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", ROOT / "storage")))
    seed: int = int(os.getenv("RANDOM_SEED", "42"))
    # chunking defaults (benchmarked in Step 4; these are starting values, not conclusions)
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "256"))        # approx tokens (word-ish units)
    chunk_overlap_pct: float = float(os.getenv("CHUNK_OVERLAP_PCT", "0.15"))
    min_chunk_chars: int = 80
    # extraction
    min_chars_per_page_for_text_pdf: int = 100
    # http
    http_timeout: int = 20
    http_max_retries: int = 4
    user_agent: str = "ai-research-paper-analyzer/2.0"
    semantic_scholar_api_key: str | None = field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY"))

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "papers.sqlite"

    @property
    def pdf_dir(self) -> Path:
        return self.storage_dir / "pdfs"

    def ensure_dirs(self) -> None:
        for p in (self.storage_dir, self.pdf_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
