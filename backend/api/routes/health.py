"""Liveness and readiness probes for containers/orchestrators. Neither forces a model load."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from flask import Blueprint, jsonify

log = logging.getLogger(__name__)
bp = Blueprint("health", __name__)
STARTED = time.time()


def resolve_index_path(p: str, storage: Path) -> Path:
    """Registry entries may be storage-relative (new) or absolute paths from another machine (old): map both to this storage."""
    q = Path(p)
    if q.exists():
        return q
    if not q.is_absolute() and (storage / q).exists():
        return storage / q
    parts = [x for x in re.split(r"[\\/]", str(p)) if x]
    return storage.joinpath(*parts[parts.index("indexes"):]) if "indexes" in parts else storage / q


@bp.get("/healthz")
def healthz():
    """Liveness: the process is up and serving requests."""
    return jsonify({"status": "ok", "uptime_s": round(time.time() - STARTED, 1)})


def readiness_checks() -> dict[str, dict]:
    from backend.config.settings import get_settings
    checks: dict[str, dict] = {}
    s = get_settings()
    try:
        from backend.api.routes.library import repo
        n = repo().count("papers")
        checks["database"] = {"ok": True, "detail": f"{n} papers"}
    except Exception as e:
        checks["database"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    try:
        probe = Path(s.storage_dir) / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        checks["storage_writable"] = {"ok": True, "detail": str(s.storage_dir.name)}
    except Exception as e:
        checks["storage_writable"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    try:
        from backend.benchmarking.model_registry import ModelRegistry
        prod = ModelRegistry(s.storage_dir / "registry").production
        if prod is None:
            checks["production_model"] = {"ok": False, "detail": "no production model in the registry; run the benchmark"}
        else:
            locs = [p for p in str(prod["index_location"]).split(";") if p]
            missing = [Path(p).name for p in locs if not resolve_index_path(p, s.storage_dir).exists()]
            checks["production_model"] = {"ok": not missing, "detail": prod["name"] + (f"; missing indexes: {missing}" if missing else "")}
    except Exception as e:
        checks["production_model"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    try:
        from backend.api.routes import retrieval as rt
        checks["retrieval_service"] = {"ok": True, "detail": ("loaded, mode " + rt._prod.service.snapshot()["mode"]) if rt._prod is not None else "not loaded yet (loads on first use)", "informational": True}
    except Exception:
        checks["retrieval_service"] = {"ok": True, "detail": "unknown", "informational": True}
    return checks


@bp.get("/readyz")
def readyz():
    """Readiness: data stores and the production model's indexes are usable. 503 lists what is missing."""
    from flask import current_app
    checks = readiness_checks()
    ready = all(c["ok"] for c in checks.values() if not c.get("informational"))
    body = {"status": "ready" if ready else "not_ready"}
    if current_app.config.get("ADMIN_API"):          # which model/index is missing is a backend detail
        body["checks"] = checks
    return jsonify(body), (200 if ready else 503)
