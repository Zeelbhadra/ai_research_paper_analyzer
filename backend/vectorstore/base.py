"""Backend-agnostic vector store interface. One store instance == one embedding model version."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int


class VectorStore(ABC):
    name = "base"

    def __init__(self, dim: int, model_version: str):
        self.dim = dim
        self.model_version = model_version

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int,
               allow: Callable[[str], bool] | None = None) -> list[Hit]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "VectorStore": ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def ids(self) -> list[str]: ...

    def vector(self, chunk_id: str):
        """Stored (normalised) vector for a chunk id, or None. Backends override."""
        return None

    def disk_bytes(self, directory: Path) -> int:
        return sum(f.stat().st_size for f in Path(directory).glob("*") if f.is_file())
