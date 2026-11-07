"""Why was this chunk retrieved?  Combines
 (1) EXACT, model-free facts about the production ranking: component ranks/scores, RRF contributions, TF-IDF term
     contributions, matched query terms, best-matching passage;
 (2) the relevance model's probability with SHAP (always) and LIME (on request) over interpretable features.
Part (2) explains a surrogate relevance model, not the embeddings and not the RRF arithmetic itself.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from backend.explainability.evidence_explainer import highlight_spans
from backend.explainability.features import FEATURES, FeatureExtractor
from backend.explainability.lime_explainer import LimeRelevanceExplainer
from backend.explainability.relevance_model import RelevanceModel
from backend.explainability.shap_explainer import ShapRelevanceExplainer
from backend.rag.citation_manager import split_sentences
from backend.rag.context_builder import content_words
from backend.retrieval.retriever import make_filter

log = logging.getLogger(__name__)


class RetrievalExplainer:
    def __init__(self, prod, model_path: Path | None = None):
        self.prod = prod
        primary = prod.service._primary
        self.supported = hasattr(primary, "sparse") and hasattr(primary, "dense")
        self.warning: str | None = None
        self.model = self.shap = self.lime = None
        if not self.supported:
            self.warning = "explanations need the hybrid production retriever"
            return
        self.extractor = FeatureExtractor(prod.chunks, primary.sparse.index, primary.dense, primary.rrf_k, primary.w_sparse, primary.w_dense)
        self.embedder = primary.dense.embedder
        path = Path(model_path or prod.storage / "models" / "relevance_model.joblib")
        if not path.exists():
            self.warning = "relevance model not trained; run `python -m backend.explainability.relevance_model train`"
            return
        rm = RelevanceModel.load(path)
        if rm.primary_version != prod.service.status.primary_version:
            self.warning = "relevance model was trained for a different production model; retrain it"
            return
        self.model = rm
        bg = np.asarray(rm.meta["background"])
        self.shap = ShapRelevanceExplainer(rm.model, rm.feature_names, bg)
        self.lime = LimeRelevanceExplainer(rm.predict_proba, rm.feature_names, bg)

    def _ensure_fresh(self) -> None:
        """Rebuild the feature extractor if the serving corpus changed (new papers were indexed)."""
        if self.supported and getattr(self, "_seen_version", 0) != getattr(self.prod, "corpus_version", 0):
            primary = self.prod.service._primary
            self.extractor = FeatureExtractor(self.prod.chunks, primary.sparse.index, primary.dense, primary.rrf_k, primary.w_sparse, primary.w_dense)
            self._seen_version = getattr(self.prod, "corpus_version", 0)

    def _best_passage(self, query: str, text: str) -> dict | None:
        sents = [s for s in split_sentences(text) if len(s.split()) >= 5]
        if not sents:
            return None
        q = self.embedder.embed_queries([query])[0]
        sims = self.embedder.embed_documents(sents) @ q
        i = int(np.argmax(sims))
        return {"text": sents[i], "cosine": round(float(sims[i]), 4)}

    def explain(self, query: str, k: int = 5, with_lime: bool = False, lime_samples: int = 1500,
                document_ids: set[str] | None = None) -> dict:
        self._ensure_fresh()
        res = self.prod.service.retrieve(query, k, make_filter(self.prod.meta, document_ids, None))
        out = {"query": query, "mode": res.mode, "fallback_active": res.fallback_active, "retriever": res.retriever,
               "retriever_kind": res.kind, "model_warning": self.warning, "results": []}
        if not self.supported:
            return out
        sc = self.extractor.corpus_scores(query)
        ids = [h.chunk_id for h in res.hits]
        X = self.extractor.extract(query, ids, sc) if ids else np.zeros((0, len(FEATURES)))
        probs = self.model.predict_proba(X) if (self.model and len(ids)) else [None] * len(ids)
        qwords = content_words(query)
        for h, x, p in zip(res.hits, X, probs):
            c = self.prod.chunks[h.chunk_id]
            sparse_rank = int(sc["sparse"][1][h.chunk_id])
            dense_rank = int(sc["dense"][1][h.chunk_id])
            e = {"rank": h.rank, "score": round(h.score, 5), "chunk_id": h.chunk_id, "title": c.paper_title, "paper_id": c.document_id,
                 "pages": [c.page_start, c.page_end], "section": c.section, "source_url": c.source_url, "text": c.text,
                 "retrieval_path": res.kind if not res.fallback_active else "sparse (TF-IDF fallback)",
                 "components": {"tfidf": {"cosine": round(float(sc["sparse"][0][h.chunk_id]), 4), "rank_in_corpus": sparse_rank},
                                "dense": {"cosine": round(float(sc["dense"][0][h.chunk_id]), 4), "rank_in_corpus": dense_rank}},
                 "tfidf_term_contributions": self.extractor.tfidf_term_contributions(query, h.chunk_id),
                 "matched_query_terms": sorted(qwords & content_words(c.text)),
                 "query_term_highlights": highlight_spans(query, c.text),
                 "best_passage": self._best_passage(query, c.text) if not res.fallback_active else None}
            if res.kind == "hybrid" and not res.fallback_active:
                pr = self.prod.service._primary
                e["rrf"] = {"sparse_part": round(pr.w_sparse / (pr.rrf_k + sparse_rank), 6), "dense_part": round(pr.w_dense / (pr.rrf_k + dense_rank), 6),
                            "formula": f"{pr.w_sparse}/({pr.rrf_k}+tfidf_rank) + {pr.w_dense}/({pr.rrf_k}+dense_rank)"}
            if self.model is not None:
                e["relevance_model"] = {"probability": round(float(p), 4), "predicted_relevant": bool(p >= self.model.threshold),
                                        "threshold": round(self.model.threshold, 3), "shap": self.shap.explain(x)}
                if with_lime:
                    e["relevance_model"]["lime"] = self.lime.explain(x, num_samples=lime_samples)
            out["results"].append(e)
        out["caveat"] = ("SHAP/LIME explain the surrogate relevance model over the listed features; they do not explain embedding "
                         "dimensions. The component ranks and RRF parts are exact facts about the production ranking.")
        return out
