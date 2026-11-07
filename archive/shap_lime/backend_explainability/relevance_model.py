"""Relevance classifier over interpretable features -- the prediction task SHAP and LIME explain.

Task: given a query and a candidate chunk from the production retriever's top-N, predict whether the chunk is
relevant (graded label >= 1 in the benchmark). Because this is a genuine binary task with ground truth,
ROC-AUC / PR-AUC / F1 / precision / recall / accuracy are meaningful here (and only here).

    python -m backend.explainability.relevance_model train
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks, records_for
from backend.benchmarking.retrieval_evaluator import classification_metrics, query_metrics
from backend.benchmarking.selection import paired_diff_ci
from backend.explainability.features import FEATURES, FeatureExtractor
from backend.config.settings import get_settings

SEED = 42


@dataclass
class RelevanceModel:
    model: object
    feature_names: list[str]
    threshold: float
    primary_version: str
    meta: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, shape (n,)."""
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "threshold": self.threshold,
                     "primary_version": self.primary_version, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "RelevanceModel":
        d = joblib.load(path)       # our own artifact
        return cls(d["model"], d["feature_names"], d["threshold"], d["primary_version"], d["meta"])


def build_rows(extractor: FeatureExtractor, retriever, ds: dict, split: str, chunks_list: list, depth: int = 20):
    X, y, meta = [], [], []
    for q in records_for(ds, split):
        rels = map_evidence_to_chunks(q["evidence"], chunks_list)
        ids = [h.chunk_id for h in retriever.retrieve(q["query"], depth)]
        X.append(extractor.extract(q["query"], ids))
        y += [int(rels.get(c, 0) >= 1) for c in ids]
        meta += [{"query_id": q["query_id"], "chunk_id": c, "pool_rank": r + 1, "grade": rels.get(c, 0)} for r, c in enumerate(ids)]
    return np.vstack(X), np.array(y), meta,


def make_gbm() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)


def make_gbm_small() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=15, random_state=SEED)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))


def best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f1))])


def rerank_ndcg(meta: list[dict], scores: np.ndarray, rels_by_q: dict) -> dict[str, float]:
    by_q: dict[str, list] = {}
    for m, s in zip(meta, scores):
        by_q.setdefault(m["query_id"], []).append((m["chunk_id"], s))
    return {q: query_metrics([c for c, _ in sorted(v, key=lambda t: -t[1])], rels_by_q[q])["nDCG@10"] for q, v in by_q.items()}


def main_train():
    from backend.retrieval.production import ProductionRetrieval
    s = get_settings()
    pr = ProductionRetrieval()
    primary = pr.service._primary
    if not hasattr(primary, "sparse") or not hasattr(primary, "dense"):
        raise SystemExit("relevance model currently requires the hybrid production retriever")
    ex = FeatureExtractor(pr.chunks, primary.sparse.index, primary.dense, primary.rrf_k, primary.w_sparse, primary.w_dense)
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    cl = list(pr.chunks.values())
    data = {sp: build_rows(ex, primary, ds, sp, cl) for sp in ("dev", "validation", "test")}
    rels = {sp: {q["query_id"]: map_evidence_to_chunks(q["evidence"], cl) for q in records_for(ds, sp)} for sp in data}
    Xd, yd, _ = data["dev"]
    Xv, yv, _ = data["validation"]
    t0 = time.perf_counter()
    cands = {"gbm_depth3": make_gbm().fit(Xd, yd), "gbm_depth2_regularized": make_gbm_small().fit(Xd, yd), "logreg": make_logreg().fit(Xd, yd)}
    train_s = time.perf_counter() - t0
    from sklearn.metrics import average_precision_score
    val_pr = {n: float(average_precision_score(yv, m.predict_proba(Xv)[:, 1])) for n, m in cands.items()}
    chosen = max(val_pr, key=val_pr.get)                      # selection rule fixed in advance: best VALIDATION PR-AUC
    model = cands[chosen]
    thr = best_f1_threshold(yv, model.predict_proba(Xv)[:, 1])  # threshold from VALIDATION only
    report = {"train_rows": len(yd), "train_pos_rate": float(yd.mean()), "train_seconds": round(train_s, 2), "chosen": chosen,
              "selection_rule": "highest validation PR-AUC among candidates", "validation_pr_auc_by_candidate": {k: round(v, 4) for k, v in val_pr.items()},
              "threshold_f1_on_validation": thr, "features": FEATURES, "splits": {}}
    for sp, (X, y, meta) in data.items():
        p_ch = model.predict_proba(X)[:, 1]
        b_nd = rerank_ndcg(meta, -np.array([m["pool_rank"] for m in meta], float), rels[sp])       # production hybrid order
        g_nd = rerank_ndcg(meta, p_ch, rels[sp])
        qids = sorted(b_nd)
        d, lo, hi = paired_diff_ci(np.array([g_nd[q] for q in qids]), np.array([b_nd[q] for q in qids]))
        report["splits"][sp] = {"rows": len(y), "pos_rate": round(float(y.mean()), 3),
                                "chosen_model": {k: round(v, 4) for k, v in classification_metrics(y, p_ch, thr).items()},
                                "all_candidates_roc_pr_auc": {n: [round(v, 4) for v in (classification_metrics(y, m.predict_proba(X)[:, 1], 0.5)["roc_auc"], classification_metrics(y, m.predict_proba(X)[:, 1], 0.5)["pr_auc"])] for n, m in cands.items()},
                                "as_reranker_nDCG@10": {"hybrid": round(float(np.mean([b_nd[q] for q in qids])), 4),
                                                        "chosen": round(float(np.mean([g_nd[q] for q in qids])), 4),
                                                        "diff": round(d, 4), "diff_ci95": [round(lo, 4), round(hi, 4)]}}
    rng = np.random.default_rng(SEED)
    background = Xd[rng.choice(len(Xd), size=min(200, len(Xd)), replace=False)]
    from backend.explainability.shap_explainer import ShapRelevanceExplainer
    report["global_shap_importance_validation"] = ShapRelevanceExplainer(model, FEATURES, background).global_importance(data["validation"][0])
    rm = RelevanceModel(model, FEATURES, thr, pr.service.status.primary_version,
                        {"kind": chosen, "trained_on": "dev", "threshold_from": "validation", "metrics": report["splits"], "seed": SEED, "background": background})
    rm.save(s.storage_dir / "models" / "relevance_model.joblib")
    (s.storage_dir / "benchmark" / "relevance_model_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["splits"], indent=1))
    print("chosen:", chosen, "| validation PR-AUC by candidate:", report["validation_pr_auc_by_candidate"])


if __name__ == "__main__":
    import sys
    {"train": main_train}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
