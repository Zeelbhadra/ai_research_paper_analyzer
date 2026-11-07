"""Shared dependencies handed to every agent (so agents stay thin and testable)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.benchmarking.model_registry import ModelRegistry
from backend.config.settings import Settings, get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.monitoring.logger import EventLog
from backend.monitoring.metrics import MetricsStore

AUTOMATION_PATH = Path(__file__).resolve().parents[1] / "config" / "automation.json"


def load_automation(path: Path | None = None) -> dict:
    return json.loads(Path(path or AUTOMATION_PATH).read_text())


@dataclass
class Services:
    settings: Settings
    repo: Repository
    mgr: EmbeddingManager
    metrics: MetricsStore
    events: EventLog
    automation: dict
    cfg_key: str = "recursive-256-15"
    production: object | None = None            # ProductionRetrieval, created lazily (loads models)
    _pr_factory: object | None = field(default=None, repr=False)

    @property
    def storage(self) -> Path:
        return self.settings.storage_dir

    @property
    def registry(self) -> ModelRegistry:
        return ModelRegistry(self.storage / "registry")      # always re-read from disk

    def get_production(self):
        if self.production is None:
            if self._pr_factory is not None:
                self.production = self._pr_factory()
            else:
                from backend.retrieval.production import ProductionRetrieval
                self.production = ProductionRetrieval(self.cfg_key, self.storage)
        return self.production

    @classmethod
    def create(cls, storage_dir: Path | None = None, cfg_key: str = "recursive-256-15", automation: dict | None = None,
               production=None, pr_factory=None) -> "Services":
        s = get_settings()
        if storage_dir:
            s.storage_dir = Path(storage_dir)
            s.ensure_dirs()
        return cls(s, Repository(s.db_path), EmbeddingManager(s.storage_dir), MetricsStore(s.storage_dir / "metrics.sqlite"),
                   EventLog(s.storage_dir / "logs" / "agent_events.jsonl"), automation or load_automation(), cfg_key, production, pr_factory)
