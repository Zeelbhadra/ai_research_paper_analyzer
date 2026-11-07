"""Interpretable (query, chunk) features for the relevance model.

Every feature is a human-readable quantity. SHAP/LIME explain a model over THESE features; they say nothing about
individual embedding dimensions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from backend.preprocessing.chunking import Chunk
from backend.rag.context_builder import content_words
from backend.rag.query_processor import SECTION_HINTS, detect_intent

FEATURES = [
    "dense_cosine",          # semantic similarity under the production embedding model
    "dense_rank_log",        # log(1 + rank among ALL chunks by dense similarity)
    "tfidf_cosine",          # lexical similarity (TF-IDF 1-2 grams)
    "tfidf_rank_log",
    "query_term_overlap",    # share of query content words present in the chunk
    "query_bigram_overlap",  # share of query content-word bigrams present in the chunk
    "title_overlap",         # share of query content words present in the paper title (metadata)
    "section_matches_intent",  # chunk section is among the sections typical for the query's intent
    "section_intro_abstract", "section_method", "section_results",   # section indicator features
    "chunk_position",        # chunk index / number of chunks in the paper (0 = start, 1 = end)
    "chunk_words",
]
_W = re.compile(r"[a-z0-9]+")


def _bigrams(text: str) -> set[tuple[str, str]]:
    w = [x for x in _W.findall(text.lower()) if len(x) > 2]
    return set(zip(w, w[1:]))


@dataclass
class FeatureExtractor:
    """Computes features for any chunks against full-corpus dense/sparse scores.
    sparse: TfidfIndex; dense: DenseRetriever (embedder + store) -- both optional (feature columns become 0)."""
    chunks: dict[str, Chunk]
    sparse: object | None
    dense: object | None
    rrf_k: int = 60
    w_sparse: float = 1.0
    w_dense: float = 0.5

    def __post_init__(self):
        self._doc_max_idx: dict[str, int] = {}
        for c in self.chunks.values():
            self._doc_max_idx[c.document_id] = max(self._doc_max_idx.get(c.document_id, 0), c.chunk_index)
        self._dense_ids = self._dense_mat = None
        if self.dense is not None:
            st = self.dense.store
            self._dense_ids = {c: i for i, c in enumerate(st.ids())}
            self._dense_mat = st.index.reconstruct_n(0, st.size)

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        r = np.empty(len(scores), dtype=np.int64)
        r[order] = np.arange(1, len(scores) + 1)
        return r

    def corpus_scores(self, query: str) -> dict:
        out: dict = {}
        if self.sparse is not None:
            s = self.sparse.score_all(query)
            out["sparse"] = (dict(zip(self.sparse.ids(), s)), dict(zip(self.sparse.ids(), self._ranks(s))))
        if self.dense is not None:
            q = self.dense.embedder.embed_queries([query])[0]
            d = self._dense_mat @ q
            ids = self.dense.store.ids()
            out["dense"] = (dict(zip(ids, d)), dict(zip(ids, self._ranks(d))))
        return out

    def extract(self, query: str, chunk_ids: list[str], scores: dict | None = None) -> np.ndarray:
        sc = scores or self.corpus_scores(query)
        intent_secs = set(SECTION_HINTS.get(detect_intent(query), []))
        qw = content_words(query)
        qb = _bigrams(query)
        rows = []
        for cid in chunk_ids:
            c = self.chunks[cid]
            ds, dr = (sc["dense"][0].get(cid, 0.0), sc["dense"][1].get(cid, len(self.chunks))) if "dense" in sc else (0.0, len(self.chunks))
            ts, tr = (sc["sparse"][0].get(cid, 0.0), sc["sparse"][1].get(cid, len(self.chunks))) if "sparse" in sc else (0.0, len(self.chunks))
            cw, tw = content_words(c.text), content_words(c.paper_title)
            rows.append([
                ds, math.log1p(dr), ts, math.log1p(tr),
                len(qw & cw) / len(qw) if qw else 0.0,
                len(qb & _bigrams(c.text)) / len(qb) if qb else 0.0,
                len(qw & tw) / len(qw) if qw else 0.0,
                float(c.section in intent_secs),
                float(c.section in ("abstract", "introduction")), float(c.section in ("methodology",)),
                float(c.section in ("results", "discussion")),
                c.chunk_index / max(self._doc_max_idx.get(c.document_id, 1), 1),
                float(len(c.text.split())),
            ])
        return np.asarray(rows, dtype=np.float64)

    # ---- exact, model-free decompositions of the production ranking ----
    def tfidf_term_contributions(self, query: str, chunk_id: str, top: int = 8) -> list[dict]:
        """Exact decomposition of the TF-IDF cosine score into per-term products q_t * d_t."""
        if self.sparse is None:
            return []
        v = self.sparse.vectorizer
        q = v.transform([query])
        pos = {c: i for i, c in enumerate(self.sparse.ids())}[chunk_id]
        d = self.sparse.matrix[pos]
        common = q.multiply(d).tocoo()
        names = v.get_feature_names_out()
        terms = sorted(((names[j], float(x)) for j, x in zip(common.col, common.data)), key=lambda t: -t[1])
        return [{"term": t, "contribution": round(x, 5)} for t, x in terms[:top]]
