import numpy as np
import pytest

from backend.embeddings.tfidf import TfidfIndex
from backend.explainability.evidence_explainer import evidence_strength, explain_answer, highlight_spans
from backend.explainability.features import FEATURES, FeatureExtractor
from backend.explainability.lime_explainer import LimeRelevanceExplainer
from backend.explainability.relevance_model import RelevanceModel, best_f1_threshold, make_gbm_small, make_logreg
from backend.explainability.shap_explainer import ShapRelevanceExplainer
from backend.preprocessing.chunking import Chunk
from backend.rag.citation_manager import analyze_answer, build_references, split_sentences
from backend.rag.context_builder import build_context

BUL = "•"


# ---------------- sentence splitting regressions (found while explaining real answers) ----------------
def test_split_keeps_trailing_citations_and_abbreviations():
    t = f"{BUL} Summary of attention. [S4]\n{BUL} Phase 3 done, e.g. SENet [5]. [S4] Next claim here. [S1][S2] Third."
    assert split_sentences(t) == [f"{BUL} Summary of attention. [S4]", f"{BUL} Phase 3 done, e.g. SENet [5]. [S4]",
                                  "Next claim here. [S1][S2]", "Third."]
    assert split_sentences("See Fig. 3 and Smith et al. 2020). It is fast! Is it? Yes.") == [
        "See Fig. 3 and Smith et al. 2020).", "It is fast!", "Is it?", "Yes."]


def test_extractive_style_bullets_count_as_cited_claims():
    ch = Chunk("A", "T", "A:0", 1, 1, "methodology",
               "The Swin Transformer computes self-attention inside shifted windows to reduce complexity.", "u", None, 0)
    items = build_context("q", [(ch, 1.0, 1)]).items
    a = analyze_answer("Extractive answer (no language model available) - relevant passages:\n"
                       f"{BUL} The Swin Transformer computes self-attention inside shifted windows to reduce complexity. [S1]", items)
    assert a.n_claims == 1 and a.n_cited == 1 and a.sentences[-1].status == "supported"


# ---------------- features ----------------
def tiny_extractor():
    texts = ["Swin transformer uses shifted windows for self attention", "graph matching solvers on benchmark datasets",
             "ride sharing platform prices and autonomous vehicles", "federated learning with privacy for computer vision tasks"]
    chunks = {f"c{i}": Chunk(f"d{i % 2}", f"Paper {i}", f"c{i}", 1, 1, "methodology" if i % 2 else "results", t, "u", None, i // 2)
              for i, t in enumerate(texts)}
    sp = TfidfIndex(min_df=1, max_df=1.0).fit(list(chunks), texts)
    return FeatureExtractor(chunks, sp, None), chunks


def test_feature_vector_shape_names_and_exact_term_decomposition():
    ex, chunks = tiny_extractor()
    q = "shifted windows self attention"
    X = ex.extract(q, ["c0", "c1"])
    assert X.shape == (2, len(FEATURES)) and "rrf_score" not in FEATURES
    f0, f1 = dict(zip(FEATURES, X[0])), dict(zip(FEATURES, X[1]))
    assert f0["tfidf_cosine"] > f1["tfidf_cosine"] and f0["query_term_overlap"] == 1.0
    assert f0["tfidf_rank_log"] == pytest.approx(np.log1p(1)) and f0["section_intro_abstract"] == 0.0
    terms = ex.tfidf_term_contributions(q, "c0", top=50)
    assert terms and all(t["contribution"] > 0 for t in terms)
    assert sum(t["contribution"] for t in terms) == pytest.approx(float(ex.sparse.score_all(q)[0]), rel=1e-3)


# ---------------- SHAP / LIME on a model with known structure ----------------
def synthetic(n=600, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    y = ((2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * r.normal(size=n)) > 0.5).astype(int)      # only features 0 and 1 matter
    return X, y


NAMES = ["strong_pos", "strong_neg", "noise_a", "noise_b"]


@pytest.mark.parametrize("make", [make_logreg, make_gbm_small])
def test_shap_additivity_and_importance_ordering(make):
    X, y = synthetic()
    m = make().fit(X, y)
    sh = ShapRelevanceExplainer(m, NAMES, X[:200])
    e = sh.explain(X[0], top=4)
    assert e["unit"].startswith("log-odds")
    if e["additivity_error"] is not None:
        assert e["additivity_error"] < 1e-4
    imp = sh.global_importance(X[:300])
    assert list(imp)[:2] == ["strong_pos", "strong_neg"] and imp["noise_a"] < 0.35 * imp["strong_pos"]
    assert e["contributions"][0]["feature"] in ("strong_pos", "strong_neg")
    x2 = X[0].copy()
    x2[0] += 2.0                                              # raising the positive driver raises its attribution
    assert sh.shap_values(x2)[0][0] > sh.shap_values(X[0])[0][0]


def test_lime_recovers_the_causal_features_locally():
    X, y = synthetic()
    m = make_logreg().fit(X, y)
    lime = LimeRelevanceExplainer(lambda Z: m.predict_proba(Z)[:, 1], NAMES, X[:300])
    x = X[np.argmin(np.abs(m.decision_function(X)))]          # near the decision boundary => informative local model
    e = lime.explain(x, top=4, num_samples=2000)
    top2 = set(sorted(e["by_feature"], key=lambda f: -abs(e["by_feature"][f]))[:2])
    assert top2 == {"strong_pos", "strong_neg"}
    assert e["by_feature"]["strong_pos"] > 0 > e["by_feature"]["strong_neg"]
    assert 0 <= e["model_prediction"] <= 1


def test_model_roundtrip_and_threshold(tmp_path):
    X, y = synthetic()
    m = make_logreg().fit(X, y)
    thr = best_f1_threshold(y, m.predict_proba(X)[:, 1])
    assert 0 < thr < 1
    rm = RelevanceModel(m, NAMES, thr, "v1", {"background": X[:10]})
    rm.save(tmp_path / "m.joblib")
    r2 = RelevanceModel.load(tmp_path / "m.joblib")
    assert np.allclose(r2.predict_proba(X[:20]), rm.predict_proba(X[:20])) and r2.primary_version == "v1"


# ---------------- evidence explainer ----------------
def test_highlight_spans_are_exact_offsets():
    passage = "The Swin Transformer computes attention inside shifted windows, which reduces complexity. It reaches 87.3 top-1 accuracy."
    sp = highlight_spans("Swin Transformer computes attention inside shifted windows and reaches 87.3 accuracy", passage)
    texts = [s["text"] for s in sp]
    assert any("shifted windows" in t for t in texts) and any("87.3" in t for t in texts)
    for s in sp:
        assert passage[s["start"]:s["end"]] == s["text"]
    assert highlight_spans("quantum annealing recipes", passage) == []


def test_evidence_strength_levels():
    assert evidence_strength("supported", 0.9) == "strong" and evidence_strength("supported", 0.55) == "moderate"
    assert evidence_strength("weak", 0.3) == "weak" and evidence_strength("unsupported", 0.1) == "none"


def test_explain_answer_distinguishes_stated_interpretation_and_unverified():
    chunks = [Chunk("A", "Paper A", "A:0", 3, 3, "methodology",
                    "The Swin Transformer computes attention inside shifted windows. It reaches 87.3 top-1 accuracy on ImageNet.", "http://a", None, 0),
              Chunk("B", "Paper B", "B:0", 7, 8, "results",
                    "Dual decomposition solvers give the tightest lower bounds on the benchmark datasets.", "http://b", None, 0)]
    items = build_context("q", [(c, 1.0, i + 1) for i, c in enumerate(chunks)], per_doc_cap=None).items
    ans = ("The Swin Transformer computes attention inside shifted windows [S1]. "
           "Dual decomposition solvers give the tightest lower bounds on the benchmark datasets [S2]. "
           "Interpretation: windowed attention is probably cheaper [S1]. "
           "Quantum annealing outperforms every classical solver by far [S2]. "
           "The weather is lovely in spring and summer.")
    a = analyze_answer(ans, items)
    resp = {"answer": ans, "answer_mode": "llm", "verification": None, "citation_quality": {"unsupported_claims": a.unsupported},
            "references": build_references(items, a, "q"),
            "sentences": [{"text": s.text, "citations": s.citations, "invalid_citations": s.invalid_citations, "status": s.status,
                           "support": s.support, "supporting_label": s.supporting_label} for s in a.sentences]}
    e = explain_answer(resp)
    kinds = {x["text"][:20]: x["kind"] for x in e["sentences"]}
    assert kinds["The Swin Transformer"] == "stated_in_source" and kinds["Interpretation: wind"] == "interpretation"
    assert kinds["Quantum annealing ou"] == "unverified" and kinds["The weather is lovel"] == "unverified"
    first = e["sentences"][0]
    assert first["evidence_strength"] in ("strong", "moderate") and first["evidence"][0]["pages"] == [3, 3]
    assert first["evidence"][0]["highlights"] and first["evidence"][0]["section"] == "methodology"
    assert e["papers"]["A"]["claims_cited"] == 1 and e["papers"]["B"]["claims_cited"] == 2
    assert abs(sum(p["share_of_cited_claims"] for p in e["papers"].values()) - 1) < 0.01
    assert e["interpretation_sentences"] and any("Quantum" in u for u in e["unsupported_claims"])
    assert next(x for x in e["sentences"] if x["text"].startswith("The weather"))["warning"].startswith("No valid citation")


def test_explain_api_validation_and_routing(monkeypatch):
    import app as flask_app
    from backend.api.routes import explainability as route

    class StubExplainer:
        model = None
        warning = "relevance model not trained"

        def explain(self, q, k, lime, document_ids=None):
            return {"query": q, "k": k, "lime": lime, "results": []}
    monkeypatch.setattr(route, "get_explainer", lambda: StubExplainer())
    c = flask_app.app.test_client()
    assert c.get("/api/explain/retrieval").status_code == 400
    assert c.get("/api/explain/retrieval?q=x&k=zz").status_code == 400
    j = c.get("/api/explain/retrieval?q=attention&k=99&lime=1").get_json()
    assert j["k"] == 10 and j["lime"] is True                           # k is clamped, lime flag parsed
    m = c.get("/api/explain/model").get_json()
    assert m["available"] is False and "not trained" in m["warning"]


def test_extractive_answer_note_warns_about_verbatim_strength():
    resp = {"answer": "x", "answer_mode": "extractive", "verification": None, "citation_quality": None, "references": [], "sentences": []}
    assert "provenance only" in explain_answer(resp)["note"]
