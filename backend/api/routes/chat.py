"""Public chat API for the assistant. Returns only reader-facing content (see backend/rag/public_view.py)."""
from __future__ import annotations

import json
import logging

from flask import Blueprint, Response, jsonify, request, stream_with_context

from backend.rag.public_view import to_public
from backend.rag.query_processor import InvalidQuery
from backend.retrieval.service import RetrievalUnavailable

bp = Blueprint("chat", __name__, url_prefix="/api/chat")
log = logging.getLogger(__name__)
BUSY = "The assistant is busy or temporarily unavailable. Please try again in a moment."


def _args():
    body = request.get_json(silent=True) or {}
    msg = body.get("message", "")
    hist = body.get("history") or []
    ids = body.get("paper_ids") or []
    if not isinstance(msg, str) or not isinstance(hist, list) or not isinstance(ids, list) or len(hist) > 20 or len(ids) > 20:
        raise InvalidQuery("Please send a message, an optional conversation history and an optional list of papers.")
    clean = []
    for m in hist:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            clean.append({"role": m["role"], "content": m["content"][:2000], "retrieval_query": str(m.get("context") or m.get("retrieval_query") or "")[:1000] or None})
    return msg, clean, [str(i) for i in ids]


def _pipeline():
    from backend.api.routes.rag import get_pipeline
    return get_pipeline()


@bp.post("")
def chat():
    try:
        msg, hist, ids = _args()
        return jsonify(to_public(_pipeline().ask(msg, hist, ids)))
    except InvalidQuery as e:
        return jsonify({"error": str(e)}), 400
    except RetrievalUnavailable:
        return jsonify({"error": BUSY}), 503
    except Exception:
        log.exception("chat failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@bp.post("/stream")
def stream():
    try:
        msg, hist, ids = _args()
        pipe = _pipeline()
    except InvalidQuery as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        log.exception("chat stream setup failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    def gen():
        first = True
        try:
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            for ev in pipe.stream(msg, hist, ids):
                if ev["type"] == "token":
                    if first and ev["text"].startswith("Extractive answer"):      # internal preamble of the passages-only path
                        first = False
                        continue
                    first = False
                    yield f"data: {json.dumps({'type': 'token', 'text': ev['text']})}\n\n"
                elif ev["type"] == "final":
                    yield f"data: {json.dumps({'type': 'final', **to_public(ev)})}\n\n"
        except RetrievalUnavailable:
            yield f"data: {json.dumps({'type': 'error', 'error': BUSY})}\n\n"
        except Exception:
            log.exception("chat stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Something went wrong. Please try again.'})}\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
