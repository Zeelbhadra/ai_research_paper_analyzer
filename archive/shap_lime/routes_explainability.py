"""Explainability API: retrieval explanations, answer evidence explanations, model summary."""
from __future__ import annotations

import json
import logging
import threading

from flask import Blueprint, jsonify, request

from backend.explainability.evidence_explainer import explain_answer
from backend.rag.query_processor import InvalidQuery
from backend.retrieval.service import RetrievalUnavailable

bp = Blueprint("explain", __name__, url_prefix="/api/explain")
log = logging.getLogger(__name__)
_exp = None
_lock = threading.Lock()


def get_explainer():
    global _exp
    with _lock:
        if _exp is None:
            from backend.api.routes.retrieval import get_production
            from backend.explainability.retrieval_explainer import RetrievalExplainer
            _exp = RetrievalExplainer(get_production())
        return _exp


@bp.get("/retrieval")
def retrieval():
    q = (request.args.get("q") or "").strip()
    if not q or len(q) > 1000:
        return jsonify({"error": "q is required (1-1000 chars)"}), 400
    try:
        k = max(1, min(int(request.args.get("k", 5)), 10))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400
    docs = set(filter(None, request.args.get("documents", "").split(","))) or None
    try:
        return jsonify(get_explainer().explain(q, k, request.args.get("lime") == "1", document_ids=docs))
    except RetrievalUnavailable as e:
        return jsonify({"error": "retrieval unavailable", "detail": str(e)}), 503
    except Exception:
        log.exception("explain retrieval failed")
        return jsonify({"error": "internal error"}), 500


@bp.post("/answer")
def answer():
    body = request.get_json(silent=True) or {}
    try:
        from backend.api.routes.rag import get_pipeline
        resp = get_pipeline().ask(body.get("question", ""), body.get("history") or [], body.get("document_ids") or [], body.get("sections") or [])
        return jsonify({"answer": resp["answer"], "explanation": explain_answer(resp), "retrieval": resp["retrieval"], "evidence_gate": resp["evidence"]})
    except InvalidQuery as e:
        return jsonify({"error": str(e)}), 400
    except RetrievalUnavailable as e:
        return jsonify({"error": "retrieval unavailable", "detail": str(e)}), 503
    except Exception:
        log.exception("explain answer failed")
        return jsonify({"error": "internal error"}), 500


@bp.get("/model")
def model_summary():
    try:
        e = get_explainer()
        report = {}
        try:
            from backend.config.settings import get_settings
            report = json.loads((get_settings().storage_dir / "benchmark" / "relevance_model_report.json").read_text())
        except Exception:
            pass
        return jsonify({"available": e.model is not None, "warning": e.warning, "features": [f for f in (e.model.feature_names if e.model else [])],
                        "report": report})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 503


@bp.post("/response")
def response_explanation():
    """Explain a RAG response the client already holds (pure function of its content; does not re-run retrieval or the LLM)."""
    body = request.get_json(silent=True) or {}
    resp = body.get("response")
    needed = ("answer", "answer_mode", "sentences", "references")
    if not isinstance(resp, dict) or any(k not in resp for k in needed):
        return jsonify({"error": "response must contain " + ", ".join(needed)}), 400
    if len(json.dumps(resp)) > 2_000_000:
        return jsonify({"error": "response too large"}), 413
    try:
        return jsonify(explain_answer(resp))
    except Exception:
        log.exception("explain response failed")
        return jsonify({"error": "could not explain this response"}), 400
