import json

import pytest

from backend.benchmarking.model_registry import ModelRegistry
from backend.benchmarking.model_selector import maybe_promote
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.retriever import Retriever
from backend.vectorstore.base import Hit

def _cand(name, score, ver, eligible=True):
    return {"name": name, "family": "x", "checkpoint": "c", "version": ver, "dim": 1, "index_location": "/i",
            "overall_score": score, "eligible": eligible, "fail_reasons": [], "metrics": {"nDCG@10": 0.5}, "config": {}}


def test_registry_history_promotion_and_guards(tmp_path):
    reg = ModelRegistry(tmp_path)
    r1 = reg.record_run({"dataset_version": "d"}, [_cand("a", 70, "va"), _cand("b", 60, "vb", eligible=False)], {"name": "a"})
    ok, _ = maybe_promote(reg, r1, "a", {}, lambda c: (True, "ok"))
    assert ok and reg.production["version"] == "va"
    with pytest.raises(ValueError):                       # ineligible candidates cannot be promoted
        reg.promote(r1, "b", "x")
    # new run: candidate c better, but validation fails => production untouched
    r2 = reg.record_run({"dataset_version": "d"}, [_cand("a", 70, "va"), _cand("c", 90, "vc")], {"name": "c"})
    ok, msg = maybe_promote(reg, r2, "c", {}, lambda c: (False, "index missing"))
    assert not ok and "production unchanged" in msg and reg.production["version"] == "va"
    # insufficient margin
    ok, msg = maybe_promote(reg, r2, "c", {"min_gain_over_current_production": 50}, lambda c: (True, "ok"))
    assert not ok and reg.production["version"] == "va"
    ok, _ = maybe_promote(reg, r2, "c", {"min_gain_over_current_production": 5}, lambda c: (True, "ok"))
    assert ok and reg.production["version"] == "vc"
    assert [p["to"] for p in reg.history()] == ["a", "c"] and reg.history()[1]["from"] == "a"
    assert not maybe_promote(reg, r2, None, {}, lambda c: (True, ""))[0]
    # persistence + history retained + csv
    reg2 = ModelRegistry(tmp_path)
    assert len(reg2.data["runs"]) == 2 and reg2.production["name"] == "c"
    assert "overall_score" in (tmp_path / "registry.csv").read_text().splitlines()[0]
    reg2.attach(r2, "heldout_test", {"c": 0.5})
    assert json.loads((tmp_path / "registry.json").read_text())["runs"][1]["attachments"]["heldout_test"] == {"c": 0.5}


class Fixed(Retriever):
    def __init__(self, ids, name): self.ids, self.name = ids, name
    def retrieve(self, query, k, allow=None): return [Hit(c, 1 / (i + 1), i + 1) for i, c in enumerate(self.ids) if allow is None or allow(c)][:k]


def test_hybrid_rrf_fusion_and_components():
    s, d = Fixed(["a", "b", "c"], "s"), Fixed(["c", "b", "d"], "d")
    h = HybridRetriever(s, d)
    out = h.retrieve("q", 4)
    assert [x.chunk_id for x in out] == ["c", "b", "a", "d"]   # c: 1/63+1/61 = .032267 > b: 2/62 = .032258 (convexity)
    assert h.last_components["b"] == {"sparse_rank": 2, "dense_rank": 2} and "dense_rank" not in h.last_components["a"]
    assert [x.chunk_id for x in HybridRetriever(s, d, 1, 0).retrieve("q", 3)] == ["a", "b", "c"]
    assert [x.chunk_id for x in h.retrieve("q", 5, allow=lambda c: c != "b")] == ["c", "a", "d"]
    with pytest.raises(ValueError):
        HybridRetriever(s, d, 0, 0)
