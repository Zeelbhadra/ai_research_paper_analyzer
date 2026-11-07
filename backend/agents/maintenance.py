"""Data-repair utilities.   python -m backend.agents.maintenance fix-titles"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.preprocessing.section_extractor import is_junk_title


def title_from_filename(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip()


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def needs_new_title(current: str, from_file: str) -> bool:
    """Junk, or a truncated first line of the real (possibly multi-line) title that the file name completes."""
    if is_junk_title(current):
        return True
    a, b = _norm(current), _norm(from_file)
    return len(b) > len(a) and b.startswith(a)


def fix_titles(repo: Repository) -> list[tuple[str, str, str]]:
    """Replace junk titles (PDF-template text, arXiv stamps, venue banners) by the file name when that looks like a title.
    Updates the paper record and the title copied into every stored chunk (all chunking configs)."""
    changes = []
    for p in repo.list_papers():
        if not p.pdf_path:
            continue
        new = title_from_filename(p.pdf_path)
        if is_junk_title(new) or new == p.title or not needs_new_title(p.title, new):
            continue
        old, p.title = p.title, new
        repo.upsert_paper(p)
        with repo._tx() as c:
            rows = c.execute("SELECT chunk_id, chunk_json FROM chunks WHERE document_id=?", (p.paper_id,)).fetchall()
            for cid, js in rows:
                d = json.loads(js)
                d["paper_title"] = new
                c.execute("UPDATE chunks SET chunk_json=? WHERE chunk_id=?", (json.dumps(d), cid))
        changes.append((p.paper_id, old, new))
    return changes


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fix-titles"
    repo = Repository(get_settings().db_path)
    if cmd == "fix-titles":
        for pid, old, new in fix_titles(repo):
            print(f"{pid}\n   {old!r}\n-> {new!r}")
