"""SQLite persistence for papers, chunks, extraction failures and pipeline jobs.

SQLite is the default because the app must run offline/without Atlas; the repository
interface is small so a Mongo implementation can be swapped in.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from backend.ingestion.models import PaperRecord
from backend.preprocessing.chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY, title TEXT, source TEXT, doi TEXT, arxiv_id TEXT, status TEXT,
  record_json TEXT NOT NULL, created_at REAL, updated_at REAL);
CREATE INDEX IF NOT EXISTS ix_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS ix_papers_arxiv ON papers(arxiv_id);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, document_id TEXT, config_key TEXT, chunk_index INTEGER,
  section TEXT, page_start INTEGER, page_end INTEGER, chunk_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(document_id, config_key);
CREATE TABLE IF NOT EXISTS extraction_failures (
  id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT, path TEXT, reason TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, status TEXT, detail TEXT, started REAL, finished REAL);
"""


class Repository:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ---- papers ----
    def upsert_paper(self, p: PaperRecord) -> None:
        now = time.time()
        with self._tx() as c:
            c.execute(
                "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(paper_id) DO UPDATE SET "
                "title=excluded.title, doi=excluded.doi, arxiv_id=excluded.arxiv_id, status=excluded.status, "
                "record_json=excluded.record_json, updated_at=excluded.updated_at",
                (p.paper_id, p.title, p.source, p.doi, p.arxiv_id, p.status, json.dumps(p.to_dict()), now, now))

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT record_json FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return PaperRecord(**json.loads(row[0])) if row else None

    def list_papers(self, status: str | None = None) -> list[PaperRecord]:
        q, a = "SELECT record_json FROM papers", ()
        if status:
            q, a = q + " WHERE status=?", (status,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY created_at", a).fetchall()
        return [PaperRecord(**json.loads(r[0])) for r in rows]

    # ---- chunks ----
    def replace_chunks(self, document_id: str, config_key: str, chunks: list[Chunk]) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM chunks WHERE document_id=? AND config_key=?", (document_id, config_key))
            c.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                [(k.chunk_id, k.document_id, config_key, k.chunk_index, k.section, k.page_start, k.page_end,
                  json.dumps(k.to_dict())) for k in chunks])

    def get_chunks(self, config_key: str, document_ids: list[str] | None = None) -> list[Chunk]:
        q, a = "SELECT chunk_json FROM chunks WHERE config_key=?", [config_key]
        if document_ids:
            q += f" AND document_id IN ({','.join('?' * len(document_ids))})"
            a += document_ids
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY document_id, chunk_index", a).fetchall()
        return [Chunk(**json.loads(r[0])) for r in rows]

    def count(self, table: str, where: str = "", args: tuple = ()) -> int:
        assert table in {"papers", "chunks", "extraction_failures", "jobs"}
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table} {where}", args).fetchone()[0]

    # ---- failures / jobs ----
    def log_failure(self, paper_id: str, path: str, reason: str) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO extraction_failures(paper_id,path,reason,ts) VALUES (?,?,?,?)",
                      (paper_id, path, reason, time.time()))

    def failures(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM extraction_failures ORDER BY ts DESC")]

    def start_job(self, kind: str) -> int:
        with self._tx() as c:
            return c.execute("INSERT INTO jobs(kind,status,started) VALUES (?,?,?)", (kind, "running", time.time())).lastrowid

    def finish_job(self, job_id: int, status: str, detail: str = "") -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET status=?, detail=?, finished=? WHERE id=?", (status, detail, time.time(), job_id))

    def jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]
