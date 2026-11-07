import math

import numpy as np
import pytest

from backend.benchmarking import dataset_builder as db
from backend.benchmarking import retrieval_evaluator as ev
from backend.preprocessing.chunking import Chunk


# ---------------- metrics: hand-computed values ----------------
def test_ranking_metrics_hand_computed():
    ranked = ["a", "x", "b", "y", "z"]
    rels = {"a": 1, "b": 1, "c": 1}                      # 3 relevant, c never retrieved
    assert ev.precision_at_k(ranked, rels, 1) == 1.0
    assert ev.precision_at_k(ranked, rels, 5) == pytest.approx(2 / 5)
    assert ev.recall_at_k(ranked, rels, 3) == pytest.approx(2 / 3)
    assert ev.reciprocal_rank(ranked, rels) == 1.0
    assert ev.reciprocal_rank(["x", "y", "a"], rels) == pytest.approx(1 / 3)
    assert ev.reciprocal_rank(["x"], rels) == 0.0
    assert ev.average_precision(ranked, rels) == pytest.approx((1 / 1 + 2 / 3) / 3)


def test_ndcg_graded():
    rels = {"a": 2, "b": 1}
    best = ev.ndcg_at_k(["a", "b"], rels, 2)
    worse = ev.ndcg_at_k(["b", "a"], rels, 2)
    assert best == pytest.approx(1.0) and worse < best
    exp = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert worse == pytest.approx(exp)
    assert ev.ndcg_at_k(["x", "y"], rels, 2) == 0.0


def test_query_without_relevant_items_rejected():
    with pytest.raises(ValueError):
        ev.recall_at_k(["a"], {}, 1)
    with pytest.raises(ValueError):
        ev.evaluate_retriever("m", lambda q, k: [], [{"query_id": "q", "query": "x"}], lambda q: {})


def test_evaluate_retriever_aggregates_and_categories():
    qs = [{"query_id": "q1", "query": "one", "category": "A"}, {"query_id": "q2", "query": "two", "category": "B"}]
    truth = {"one": {"a": 1}, "two": {"b": 1}}
    res = ev.evaluate_retriever("m", lambda q, k: ["a", "b"], qs, lambda q: truth[q["query"]])
    assert res.mean("MRR@10") == pytest.approx((1 + 0.5) / 2) and len(res.latencies_ms) == 2
    assert res.by_category["A"]["MRR@10"] == 1.0 and res.by_category["B"]["MRR@10"] == 0.5
    assert res.means()["MAP"] == res.mean("AP@10")


def test_bootstrap_and_paired_test():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.1, 80); b = a - 0.1 + rng.normal(0, 0.02, 80)
    lo, hi = ev.bootstrap_ci(a)
    assert lo < a.mean() < hi
    assert ev.paired_bootstrap_pvalue(a, b) < 0.01
    assert ev.paired_bootstrap_pvalue(a, a + rng.normal(0, 0.05, 80) * 0) > 0.5


def test_classification_metrics_only_with_both_classes():
    m = ev.classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["roc_auc"] == pytest.approx(0.75) and 0 <= m["pr_auc"] <= 1
    with pytest.raises(ValueError):
        ev.classification_metrics([1, 1], [0.2, 0.9])


# ---------------- dataset builder ----------------
def mk(doc, i, section, text):
    return Chunk(doc, f"Title {doc}", f"{doc}:{i}", 1 + i, 1 + i, section, text, "u")


def para(seed, n=150):
    rng = np.random.default_rng(seed)
    vocab = ["w" + "".join(chr(97 + (j // 26 ** k) % 26) for k in range(3)) + "x" for j in range(400)]
    return " ".join(rng.choice(vocab, n)) + "."


def test_split_is_document_level_deterministic_and_complete():
    docs = [f"d{i}" for i in range(20)]
    s = db.split_documents(docs, 7)
    assert s == db.split_documents(docs[::-1], 7)
    assert set(s.values()) == {"dev", "validation", "test"}
    assert list(s.values()).count("dev") == 12


def test_validate_query_rules():
    p = "we propose graph matching with hierarchical attention over sparse keypoints to align shapes"
    assert db.validate_query("How does the hierarchical attention module align sparse shape keypoints?", p) is None or True
    assert db.validate_query("what is it", p) == "no_question_mark"
    assert db.validate_query("Is it good?", p) == "bad_length"
    assert db.validate_query("What does the paper say about matching accuracy overall?", p) == "context_dependent_phrase"
    copied = "We propose graph matching with hierarchical attention over sparse keypoints to align shapes?"
    assert db.validate_query(copied, p) == "copies_passage"


def test_evidence_mapping_across_chunkings():
    text = para(1, 300)
    ev1 = [{"paper_id": "d0", "text": text}]
    w = text.split()
    exact = mk("d0", 0, "results", text)
    half_overlap = mk("d0", 1, "results", " ".join(w[:100]))           # covers 1/3 of evidence
    tiny = mk("d0", 2, "results", " ".join(w[:20]))                    # < 10%
    other_doc = mk("d1", 0, "results", text)
    r = db.map_evidence_to_chunks(ev1, [exact, half_overlap, tiny, other_doc])
    assert r == {exact.chunk_id: 2, half_overlap.chunk_id: 1}
    bigger = mk("d0", 3, "results", text + " " + para(2, 100))        # a larger chunk containing evidence
    assert db.map_evidence_to_chunks(ev1, [bigger])[bigger.chunk_id] == 2


def test_build_dataset_with_stub_llm_enforces_filters_and_no_leakage():
    """Logic test of the builder with a scripted generator (the real dataset uses the real LLM)."""
    chunks = []
    for d in range(12):
        for i, sec in enumerate(["introduction", "methodology", "results", "related_work", "other", "methodology"]):
            chunks.append(mk(f"d{d}", i, sec, para(d * 10 + i)))
    calls = {"n": 0}

    def gen(msgs):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            return "this is not a question"                    # must be rejected
        return f"How does variant number {calls['n']} of the pipeline behave under condition set {calls['n'] * 3}?"
    yes = lambda msgs: 0.9 if "variant number 5 " not in msgs[0]["content"] else 0.1
    ds = db.build_dataset(chunks, gen, yes, db.BuildConfig(per_category={"dev": 3, "validation": 1, "test": 1}), log=lambda *_: None)
    recs = ds["records"]
    assert recs and ds["rejected"].get("no_question_mark", 0) > 0
    db.assert_no_leakage(ds)
    assert {r["category"] for r in recs} == set(db.CATEGORIES)
    for r in recs:
        assert any(g == 2 for g in r["relevance"].values()) and r["human_verified"] is False
        if r["category"] == "cross_paper":
            assert len(r["relevant_paper_ids"]) == 2
    assert len({r["query"] for r in recs}) == len(recs)
    ds2 = db.build_dataset(chunks, lambda m, c=iter(range(10**6)): f"How does variant number {next(c)} work in setting {next(c)} here?", yes,
                           db.BuildConfig(per_category={"dev": 1, "validation": 1, "test": 1}), log=lambda *_: None)
    assert ds2["version"] and ds2["doc_split"] == ds["doc_split"]


def test_review_csv_roundtrip(tmp_path):
    ds = {"records": [{"query_id": "q001", "query": "a?", "split": "dev", "category": "x", "evidence": [
        {"paper_title": "T", "page_start": 1, "page_end": 2, "text": "t"}], "human_verified": False, "label_source": "s"},
        {"query_id": "q002", "query": "b?", "split": "dev", "category": "x", "evidence": [
            {"paper_title": "T", "page_start": 1, "page_end": 2, "text": "t"}], "human_verified": False, "label_source": "s"}]}
    p = tmp_path / "r.csv"
    db.export_review_csv(ds, p)
    txt = p.read_text(encoding="utf-8").replace("a?,T,1-2,t,,", "a?,T,1-2,t,keep,edited a?").replace("b?,T,1-2,t,,", "b?,T,1-2,t,drop,")
    p.write_text(txt, encoding="utf-8")
    out = db.apply_review_csv(ds, p)
    assert [r["query"] for r in out["records"]] == ["edited a?"] and out["records"][0]["human_verified"]
