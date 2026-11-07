"""RAG API: POST /api/rag/ask (JSON) and POST /api/rag/stream (server-sent events)."""
from __future__ import annotations

import json
import logging
import threading

from flask import Blueprint, Response, jsonify, request, stream_with_context

from backend.rag.query_processor import InvalidQuery
from backend.retrieval.service import RetrievalUnavailable

bp = Blueprint("rag", __name__, url_prefix="/api/rag")
log = logging.getLogger(__name__)
_pipe = None
_lock = threading.Lock()


def get_pipeline():
    global _pipe
    with _lock:
        if _pipe is None:
            from backend.api.routes.retrieval import get_production
            from backend.rag.llm import default_chain
            from backend.rag.pipeline import RAGConfig, RAGPipeline
            cfg = RAGConfig.load()
            reranker = None
            if cfg.use_reranker:
                from backend.retrieval.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
            _pipe = RAGPipeline(get_production(), default_chain(), reranker, cfg)
        return _pipe


def _args():
    body = request.get_json(silent=True) or {}
    q = body.get("question", "")
    hist = body.get("history") or []
    if not isinstance(hist, list) or len(hist) > 40:
        raise InvalidQuery("history must be a list of at most 40 messages")
    docs, secs = body.get("document_ids") or [], body.get("sections") or []
    if not isinstance(docs, list) or not isinstance(secs, list):
        raise InvalidQuery("document_ids and sections must be lists")
    return q, hist, [str(d) for d in docs][:50], [str(s) for s in secs][:20]


@bp.post("/ask")
def ask():
    try:
        q, hist, docs, secs = _args()
        return jsonify(get_pipeline().ask(q, hist, docs, secs))
    except InvalidQuery as e:
        return jsonify({"error": str(e)}), 400
    except RetrievalUnavailable as e:
        return jsonify({"error": "retrieval unavailable", "detail": str(e)}), 503
    except Exception:
        log.exception("rag ask failed")
        return jsonify({"error": "internal error"}), 500


@bp.post("/stream")
def stream():
    try:
        q, hist, docs, secs = _args()
        pipe = get_pipeline()
    except InvalidQuery as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        log.exception("rag stream setup failed")
        return jsonify({"error": "internal error"}), 500

    def gen():
        try:
            for ev in pipe.stream(q, hist, docs, secs):
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:
            log.exception("rag stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': type(e).__name__})}\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
