import json

import pytest

from backend.monitoring.health_checks import probe
from backend.monitoring.logger import EventLog
from backend.retrieval.retriever import Retriever
from backend.retrieval.service import RetrievalService, RetrievalUnavailable, ServiceConfig
from backend.vectorstore.base import Hit


class Toggle(Retriever):
    """Retriever whose failure mode can be switched at will."""
    kind = "dense"

    def __init__(self, name, ids, kind="dense"):
        self.name, self.ids, self.kind = name, ids, kind
        self.fail = None          # None | 'retrieve' | 'health'
        self.calls = 0

    def retrieve(self, query, k, allow=None):
        self.calls += 1
        if self.fail == "retrieve":
            raise RuntimeError("embedding backend down")
        return [Hit(c, 1.0 / (i + 1), i + 1) for i, c in enumerate(self.ids) if allow is None or allow(c)][:k]

    def health(self):
        return self.fail != "health"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(primary=None, fallback=None, **cfg):
    clock = Clock()
    p = primary or Toggle("hybrid", ["a", "b", "c"], "hybrid")
    f = fallback or Toggle("tfidf", ["x", "y"], "sparse")
    ev = EventLog()
    svc = RetrievalService(lambda: p, lambda: f, "hybrid", "v1", "tfidf", ServiceConfig(**cfg), ev, clock)
    return svc, p, f, clock, ev


def test_normal_operation_uses_primary():
    svc, p, f, *_ = make()
    r = svc.retrieve("q", 2)
    assert r.mode == "primary" and not r.fallback_active and [h.chunk_id for h in r.hits] == ["a", "b"] and f.calls == 0


def test_failing_request_is_served_by_fallback_then_mode_switches_at_threshold():
    svc, p, f, clock, ev = make(failure_threshold=2)
    p.fail = "retrieve"
    r1 = svc.retrieve("q", 2)                    # 1st failure: served by fallback, mode still primary
    assert r1.mode == "fallback" and r1.hits and "embedding backend down" in r1.error
    assert not svc.snapshot()["fallback_active"]
    svc.retrieve("q", 2)                         # 2nd consecutive failure: switch
    assert svc.snapshot()["fallback_active"] and svc.snapshot()["mode"] == "fallback"
    calls = p.calls
    r3 = svc.retrieve("q", 2)
    assert r3.fallback_active and r3.retriever == "tfidf" and p.calls == calls      # primary no longer hammered
    assert [e["kind"] for e in ev.tail()].count("fallback_activated") == 1
    assert svc.snapshot()["fallback_requests"] == 3 and svc.snapshot()["total_failures"] == 2


def test_success_resets_consecutive_failures():
    svc, p, *_ = make(failure_threshold=2)
    p.fail = "retrieve"
    svc.retrieve("q")
    p.fail = None
    svc.retrieve("q")
    p.fail = "retrieve"
    svc.retrieve("q")
    assert svc.snapshot()["mode"] == "primary"


def test_recovery_needs_consecutive_healthy_probes_and_respects_cooldown():
    svc, p, f, clock, ev = make(failure_threshold=1, recovery_checks=2, cooldown_base_s=10)
    p.fail = "retrieve"
    svc.retrieve("q")
    assert svc.snapshot()["fallback_active"]
    assert svc.check_health()["detail"] == "cooling down"           # too early
    clock.t += 11
    r = svc.check_health()                                          # still broken -> failed probe, backoff doubles
    assert not r["ok"] and svc.snapshot()["fallback_active"]
    assert svc.snapshot()["next_recovery_at"] - clock.t == pytest.approx(10 * 2 ** 1)
    p.fail = None
    clock.t += 21
    assert svc.check_health()["mode"] == "fallback"                 # 1 healthy probe is not enough
    clock.t += 11
    out = svc.check_health()
    assert out["ok"] and out["mode"] == "primary" and not svc.snapshot()["fallback_active"]
    assert svc.retrieve("q").mode == "primary"
    kinds = [e["kind"] for e in ev.tail()]
    assert "recovery_probe_failed" in kinds and "primary_restored" in kinds


def test_flapping_primary_resets_recovery_streak():
    svc, p, f, clock, _ = make(failure_threshold=1, recovery_checks=2, cooldown_base_s=1)
    p.fail = "retrieve"
    svc.retrieve("q")
    p.fail = None
    clock.t += 5
    svc.check_health()
    assert svc.snapshot()["healthy_probes"] == 1
    p.fail = "health"
    clock.t += 5
    svc.check_health()
    assert svc.snapshot()["healthy_probes"] == 0 and svc.snapshot()["fallback_active"]


def test_primary_load_failure_starts_in_fallback_and_can_recover():
    state = {"ok": False}
    p = Toggle("hybrid", ["a"], "hybrid")
    clock, f = Clock(), Toggle("tfidf", ["x"], "sparse")

    def factory():
        if not state["ok"]:
            raise OSError("index file missing")
        return p
    svc = RetrievalService(factory, lambda: f, "hybrid", "v1", "tfidf",
                           ServiceConfig(recovery_checks=1, cooldown_base_s=1), EventLog(), clock)
    assert svc.snapshot()["fallback_active"] and "index file missing" in svc.snapshot()["reason"]
    assert svc.retrieve("q").hits[0].chunk_id == "x"
    state["ok"] = True
    clock.t += 5
    assert svc.check_health()["mode"] == "primary" and svc.retrieve("q").retriever == "hybrid"


def test_health_probe_failure_in_primary_mode_counts_as_failure():
    svc, p, *_ = make(failure_threshold=1)
    p.fail = "health"
    assert not svc.check_health()["ok"] and svc.snapshot()["fallback_active"]


def test_no_independent_fallback_raises_unavailable():
    p = Toggle("tfidf", ["a"], "sparse")
    svc = RetrievalService(lambda: p, None, "tfidf", "v1", None, ServiceConfig(), EventLog(), Clock())
    assert svc.retrieve("q").hits
    p.fail = "retrieve"
    with pytest.raises(RetrievalUnavailable):
        svc.retrieve("q")
    assert not svc.snapshot()["fallback_active"]


def test_both_down_raises():
    svc, p, f, *_ = make(failure_threshold=1)
    p.fail = f.fail = "retrieve"
    with pytest.raises(Exception):
        svc.retrieve("q")


def test_filters_pass_through_in_both_modes():
    svc, p, f, *_ = make(failure_threshold=1)
    assert [h.chunk_id for h in svc.retrieve("q", 5, lambda c: c != "a").hits] == ["b", "c"]
    p.fail = "retrieve"
    assert [h.chunk_id for h in svc.retrieve("q", 5, lambda c: c != "x").hits] == ["y"]


def test_safe_model_switch_keeps_working_model_on_failure():
    svc, p, f, clock, ev = make()
    bad = Toggle("bge", ["z"])
    bad.fail = "health"
    ok, msg = svc.switch_primary(lambda: bad, "bge", "v2")
    assert not ok and svc.snapshot()["primary"] == "hybrid" and svc.retrieve("q").retriever == "hybrid"

    def boom():
        raise OSError("no index")
    ok, msg = svc.switch_primary(boom, "e5", "v3")
    assert not ok and svc.snapshot()["primary_version"] == "v1"
    good = Toggle("bge", ["z", "w"])
    ok, msg = svc.switch_primary(lambda: good, "bge", "v2")
    assert ok and svc.snapshot()["primary_version"] == "v2" and svc.retrieve("q").retriever == "bge"
    kinds = [e["kind"] for e in ev.tail()]
    assert kinds.count("model_switch_failed") == 2 and "model_switched" in kinds


def test_switch_while_in_fallback_restores_service():
    svc, p, f, clock, _ = make(failure_threshold=1)
    p.fail = "retrieve"
    svc.retrieve("q")
    assert svc.snapshot()["fallback_active"]
    ok, _ = svc.switch_primary(lambda: Toggle("e5", ["m"]), "e5", "v9")
    assert ok and not svc.snapshot()["fallback_active"] and svc.retrieve("q").retriever == "e5"


def test_probe_and_event_log(tmp_path):
    assert probe(Toggle("t", ["a"])).ok
    assert not probe(Toggle("t", [])).ok                                       # smoke query returns nothing
    bad = Toggle("t", ["a"])
    bad.fail = "retrieve"
    assert "RuntimeError" in probe(bad).detail
    log = EventLog(tmp_path / "e.jsonl", keep=2)
    for i in range(3):
        log.emit("k", i=i)
    assert [e["i"] for e in log.tail()] == [1, 2]
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["i"] == 0


def test_api_status_search_and_error_mapping(monkeypatch):
    import app as flask_app
    from backend.api.routes import retrieval as route
    svc, p, f, *_ = make(failure_threshold=1)

    class Stub:
        service = svc
        events = EventLog()

        def search(self, q, k, d, s):
            r = self.service.retrieve(q, k)
            return {"query": q, "mode": r.mode, "fallback_active": r.fallback_active, "results": [h.chunk_id for h in r.hits]}

        def status(self):
            return self.service.snapshot()

        def sync_with_registry(self):
            return False, "noop"
    stub = Stub()
    monkeypatch.setattr(route, "get_production", lambda: stub)
    c = flask_app.app.test_client()
    assert c.get("/api/retrieval/search").status_code == 400
    assert c.get("/api/retrieval/search?q=x&k=abc").status_code == 400
    assert c.get("/api/retrieval/search?q=x").get_json()["mode"] == "primary"
    p.fail = "retrieve"
    j = c.get("/api/retrieval/search?q=x").get_json()
    assert j["fallback_active"] and c.get("/api/retrieval/status").get_json()["fallback_active"]
    p.fail = None
    assert c.post("/api/retrieval/health-check?force=1").get_json()["mode"] == "fallback"     # needs 2 probes
    assert c.post("/api/retrieval/health-check?force=1").get_json()["mode"] == "primary"
    # both retrievers down -> 503, not a crash
    svc2, p2, f2, *_ = make(failure_threshold=1, primary=Toggle("h", ["a"]), fallback=Toggle("t", ["b"]))
    p2.fail = f2.fail = "retrieve"
    stub.service = svc2
    r = c.get("/api/retrieval/search?q=x")
    assert r.status_code in (500, 503)


def test_ensure_indexes_builds_tfidf_only_when_missing(tmp_path, repo):
    from backend.embeddings.embedding_manager import EmbeddingManager
    from backend.preprocessing.chunking import Chunk
    from backend.retrieval.production import components_of, ensure_indexes
    assert components_of("hybrid:e5:1.0:0.5") == ["tfidf", "e5"] and components_of("tfidf") == ["tfidf"]
    ch = [Chunk("d", "T", f"c{i}", 1, 1, "results", t, "u")
          for i, t in enumerate(["graph matching solvers", "ride sharing demand pricing", "vision transformer attention",
                                 "federated learning privacy", "bayesian optimisation of hyperparameters"])]
    repo.replace_chunks("d", "cfg", ch)
    mgr = EmbeddingManager(tmp_path)
    assert ensure_indexes("tfidf", repo, mgr, "cfg") == ["tfidf"]
    assert ensure_indexes("tfidf", repo, mgr, "cfg") == []
