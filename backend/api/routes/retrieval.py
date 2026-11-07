"""Retrieval API: search, status (fallback indicator for the UI), health/recovery, registry sync."""
from __future__ import annotations

import logging
import threading

from flask import Blueprint, jsonify, request

from backend.retrieval.service import RetrievalUnavailable

bp = Blueprint("retrieval", __name__, url_prefix="/api/retrieval")
log = logging.getLogger(__name__)
_prod = None
_lock = threading.Lock()


def get_production():
    """Lazy singleton so importing the app never loads models."""
    global _prod
    with _lock:
        if _prod is None:
            from backend.retrieval.production import ProductionRetrieval
            _prod = ProductionRetrieval()
        return _prod


@bp.get("/search")
def search():
    q = (request.args.get("q") or "").strip()
    if not q or len(q) > 1000:
        return jsonify({"error": "q is required (1-1000 chars)"}), 400
    try:
        k = max(1, min(int(request.args.get("k", 10)), 50))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400
    docs = set(filter(None, request.args.get("documents", "").split(","))) or None
    secs = set(filter(None, request.args.get("sections", "").split(","))) or None
    try:
        return jsonify(get_production().search(q, k, docs, secs))
    except RetrievalUnavailable as e:
        return jsonify({"error": "retrieval unavailable", "detail": str(e)}), 503
    except Exception:
        log.exception("search failed")
        return jsonify({"error": "internal error"}), 500


@bp.get("/status")
def status():
    try:
        return jsonify(get_production().status())
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@bp.post("/health-check")
def health_check():
    return jsonify(get_production().service.check_health(force=request.args.get("force") == "1"))


@bp.post("/sync")
def sync():
    ok, msg = get_production().sync_with_registry()
    return jsonify({"switched": ok, "message": msg})


@bp.get("/events")
def events():
    return jsonify(get_production().events.tail(100))
