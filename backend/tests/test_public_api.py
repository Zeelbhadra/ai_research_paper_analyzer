"""Reader-facing API: admin routes are off by default and chat responses carry no backend details."""
import json
import re

import pytest

from backend.rag.public_view import renumber, to_public
from backend.retrieval.service import RetrievalUnavailable
from backend.tests.test_rag import Stub, pipe

# words that would reveal how the system works internally
FORBIDDEN = re.compile(r"tf-?idf|hybrid|sbert|\be5\b|\bbge\b|faiss|embedding|cosine|fallback|retriev|nDCG|latency|\brank\b|\bscore\b|prompt_tokens|provider|openai|gate|threshold|token", re.I)


@pytest.fixture
def pub():
    from app import create_app
    return create_app(admin=False)


def test_admin_routes_exist_only_when_enabled(pub):
    from app import create_app
    c = pub.test_client()
    for path in ("/api/dashboard", "/api/benchmarks/overview", "/api/pipeline/status", "/api/pipeline/jobs", "/api/retrieval/status", "/api/retrieval/search?q=x",
                 "/api/explain/model", "/api/explain/retrieval?q=x"):
        assert c.get(path).status_code == 404, path
    assert c.post("/api/rag/ask", json={"question": "hello there"}).status_code == 404
    assert c.post("/api/pipeline/run", json={"workflow": "report"}).status_code == 404
    admin = create_app(admin=True).test_client()
    assert admin.get("/api/dashboard").status_code == 200
    public_rules = {r.rule for r in pub.url_map.iter_rules()}
    assert {"/", "/api/search", "/api/analyze", "/api/similar", "/api/chat", "/api/chat/stream", "/api/library/papers", "/healthz", "/readyz"} <= public_rules


def test_public_readyz_hides_internal_checks(pub):
    r = pub.test_client().get("/readyz")
    body = r.get_json()
    assert set(body) == {"status"} and r.status_code in (200, 503)


def test_index_page_has_only_reader_pages_and_no_backend_terms(pub):
    html = pub.test_client().get("/").get_data(as_text=True)
    for page in ("discover", "library", "assistant"):
        assert f'data-route="{page}"' in html
    for gone in ("dashboard", "benchmarks", "explain", "pipeline"):
        assert f'data-route="{gone}"' not in html and f"view-{gone}" not in html
    visible = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html)
    visible = re.sub(r"<[^>]+>", " ", visible)
    assert not FORBIDDEN.search(visible), FORBIDDEN.search(visible).group(0)
    for js in ("view_dashboard", "view_benchmarks", "view_explain", "view_pipeline", "charts.js"):
        assert js not in html


def test_renumber():
    assert renumber("A [S1][S3] and [S2, S4].") == "A [1][3] and [2][4]."


def test_public_response_has_only_reader_facing_fields_and_numbered_sources():
    r = pipe(Stub("The Swin Transformer computes attention inside shifted windows which reduces the complexity of vision models [S1]. Windows are shifted between consecutive layers to connect neighbouring windows [S2].")
             ).ask("How does the Swin Transformer reduce complexity with shifted windows?", document_ids=["A"])
    pub = to_public(r)
    assert set(pub) == {"answer", "kind", "sentences", "sources", "notes", "suggestions", "context"}
    assert pub["kind"] == "answer" and "[1]" in pub["answer"] and "[S1]" not in pub["answer"] and pub["suggestions"]
    src = pub["sources"][0]
    assert set(src) == {"n", "cited", "paper_id", "title", "pages", "section", "url", "quote", "text", "highlights"} and src["pages"][0] >= 1
    assert all(set(s) == {"text", "kind", "sources"} for s in pub["sentences"])
    blob = json.dumps({k: v for k, v in pub.items() if k not in ("sources", "context")})          # source text itself is paper content, not backend detail
    assert not FORBIDDEN.search(blob), FORBIDDEN.search(blob).group(0)


def test_public_notes_for_passages_and_rejected_drafts_and_refusals():
    ext = to_public(pipe(None).ask("How does the Swin Transformer compute attention with shifted windows?"))
    assert ext["kind"] == "passages" and not any(s["text"].lower().startswith("extractive answer") for s in ext["sentences"])
    assert ext["notes"] == ["Here are the most relevant passages from the paper."] and "extractive" not in ext["answer"].lower()
    bad = to_public(pipe(Stub("The Swin Transformer partitions images into windows. It then does something remarkable everywhere.")
                         ).ask("How does the Swin Transformer compute attention with shifted windows?"))
    assert bad["kind"] == "passages" and "could not confirm" in bad["notes"][0] and "partitions images" not in json.dumps(bad)      # the rejected draft is never shown
    none = to_public(pipe(Stub("x [S1]"), min_cov=0.9).ask("What is the recipe for sourdough bread baking?"))
    assert none["kind"] == "none" and none["sources"] == [] or none["kind"] == "none"
    assert "could not find enough information" in none["answer"] and none["suggestions"] == []


# ---------------- chat routes ----------------
def test_chat_validation_and_friendly_errors(pub, monkeypatch):
    from backend.api.routes import chat
    c = pub.test_client()
    assert c.post("/api/chat", json={"message": ""}).status_code == 400
    assert c.post("/api/chat", json={"message": "hello there", "history": "x"}).status_code == 400
    assert c.post("/api/chat", json={"message": "hello there", "paper_ids": "x"}).status_code == 400
    monkeypatch.setattr(chat, "_pipeline", lambda: type("P", (), {"ask": lambda self, *a, **k: (_ for _ in ()).throw(RetrievalUnavailable("down: primary and fallback"))})())
    r = c.post("/api/chat", json={"message": "How are windows shifted?"})
    assert r.status_code == 503 and "try again" in r.get_json()["error"].lower() and not FORBIDDEN.search(r.get_data(as_text=True))
    monkeypatch.setattr(chat, "_pipeline", lambda: type("P", (), {"ask": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal path C:\\x"))})())
    r = c.post("/api/chat", json={"message": "How are windows shifted?"})
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)


def test_chat_answer_and_stream_never_leak_internals(pub, monkeypatch):
    from backend.api.routes import chat
    p = pipe(Stub("The Swin Transformer computes attention inside shifted windows, which reduces complexity from quadratic to linear in image size [S1]."))
    monkeypatch.setattr(chat, "_pipeline", lambda: p)
    c = pub.test_client()
    d = c.post("/api/chat", json={"message": "How are the windows shifted between layers?", "paper_ids": ["A"],
                                  "history": [{"role": "user", "content": "earlier", "context": "How does Swin work?"}, {"role": "assistant", "content": "earlier answer"}]}).get_json()
    assert d["kind"] == "answer" and d["sources"] and "[1]" in d["answer"]
    body = c.post("/api/chat/stream", json={"message": "How are the windows shifted between layers?"}).get_data(as_text=True)
    events = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]
    assert events[0]["type"] == "start" and events[-1]["type"] == "final" and events[-1]["sources"]
    assert all(set(e) <= {"type", "text"} for e in events if e["type"] != "final")                    # progress events carry nothing but text
    assert not FORBIDDEN.search(" ".join(e.get("text", "") for e in events if e["type"] != "final"))
    # passages-only path: the internal preamble must not be streamed
    monkeypatch.setattr(chat, "_pipeline", lambda: pipe(None))
    body = c.post("/api/chat/stream", json={"message": "How are the windows shifted between layers?"}).get_data(as_text=True)
    events = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]
    assert not any(e["type"] == "token" and "Extractive answer" in e["text"] for e in events) and events[-1]["kind"] == "passages"
