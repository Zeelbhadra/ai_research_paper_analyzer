"""Real explainability run + SHAP/LIME consistency study.   python -m backend.benchmarking.explain_demo"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from backend.benchmarking.dataset_builder import load_dataset, records_for
from backend.config.settings import get_settings
from backend.explainability.evidence_explainer import explain_answer
from backend.explainability.retrieval_explainer import RetrievalExplainer
from backend.rag.pipeline import RAGConfig, RAGPipeline
from backend.retrieval.production import ProductionRetrieval


def main(n_agree: int = 40):
    s = get_settings()
    pr = ProductionRetrieval()
    ex = RetrievalExplainer(pr)
    assert ex.model is not None, ex.warning
    out = {}
    q = "How does the Swin Transformer reduce the computational complexity of self-attention?"
    r = ex.explain(q, 3, with_lime=True)
    out["retrieval_example"] = {"query": q, "mode": r["mode"], "results": [
        {"rank": e["rank"], "title": e["title"][:55], "pages": e["pages"], "section": e["section"], "score": e["score"],
         "components": e["components"], "rrf": e.get("rrf"), "top_tfidf_terms": e["tfidf_term_contributions"][:4],
         "matched_terms": e["matched_query_terms"], "best_passage": e["best_passage"],
         "p_relevant": e["relevance_model"]["probability"],
         "shap_top": [(c["feature"], c["shap"]) for c in e["relevance_model"]["shap"]["contributions"][:4]],
         "shap_additivity_error": e["relevance_model"]["shap"]["additivity_error"],
         "lime_top": e["relevance_model"]["lime"]["weights"][:4], "lime_r2": e["relevance_model"]["lime"]["local_model_r2"]} for e in r["results"]]}

    # ---- SHAP vs LIME on held-out candidates ----
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    rng = np.random.default_rng(42)
    rows = []
    for sp in ("validation", "test"):
        for rec in records_for(ds, sp):
            ids = [h.chunk_id for h in pr.service.retrieve(rec["query"], 20).hits]
            X = ex.extractor.extract(rec["query"], ids)
            rows += [X[i] for i in rng.choice(len(ids), size=min(2, len(ids)), replace=False)]
    rows = [rows[i] for i in rng.choice(len(rows), size=min(n_agree, len(rows)), replace=False)]
    overlap3, rhos, add_err, r2s = [], [], [], []
    names = ex.model.feature_names
    for x in rows:
        sh = ex.shap.explain(x, top=len(names))
        add_err.append(sh["additivity_error"])
        li = ex.lime.explain(x, top=len(names), num_samples=1500)
        sv = {c["feature"]: abs(c["shap"]) for c in sh["contributions"]}
        lv = {f: abs(li["by_feature"].get(f, 0.0)) for f in names}
        a, b = np.array([sv[f] for f in names]), np.array([lv[f] for f in names])
        overlap3.append(len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) / 3)
        rhos.append(spearmanr(a, b)[0])
        r2s.append(li["local_model_r2"])
    out["shap_vs_lime"] = {"n_instances": len(rows), "mean_top3_feature_overlap": round(float(np.mean(overlap3)), 3),
                           "mean_spearman_abs_attribution": round(float(np.nanmean(rhos)), 3),
                           "max_shap_additivity_error": float(np.max(add_err)), "mean_lime_local_r2": round(float(np.mean(r2s)), 3),
                           "chance_top3_overlap": round(3 / len(names), 3)}

    # ---- answer evidence explanation (extractive path: no LLM available) ----
    pipe = RAGPipeline(pr, None, None, RAGConfig.load())
    resp = pipe.ask(q)
    exp = explain_answer(resp)
    out["answer_explanation"] = {"answer_mode": exp["answer_mode"], "papers": exp["papers"], "unsupported_claims": exp["unsupported_claims"],
                                 "sentences": [{"text": x["text"][:140], "kind": x["kind"], "strength": x["evidence_strength"],
                                                "evidence": [(e["title"][:40], e["pages"], e["section"], [h["text"] for h in e["highlights"]][:3]) for e in x["evidence"]],
                                                "warning": x["warning"]} for x in exp["sentences"]]}
    (s.storage_dir / "benchmark" / "explain_demo.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
