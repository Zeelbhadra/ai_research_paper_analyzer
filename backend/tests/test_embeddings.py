import numpy as np
import pytest

from backend.embeddings.base import (BGEEmbedder, DenseEmbedder, E5Embedder, EmbeddingError, SBERTEmbedder,
                                     make_embedder)
from backend.embeddings.embedding_manager import EmbeddingCache, EmbeddingManager
from backend.embeddings.tfidf import TfidfIndex
from backend.preprocessing.chunking import Chunk
from backend.retrieval.retriever import DenseRetriever, SparseRetriever, make_filter
from backend.vectorstore.faiss_store import FaissStore


class FakeEmbedder(DenseEmbedder):
    """Deterministic hashed bag-of-words embedder; counts encode calls."""
    family, checkpoint, dim = "fake", "fake/bow", 64

    def __init__(self, qp="", dp="", bad=None):
        self.qp, self.dp, self.bad, self.calls = qp, dp, bad, 0

    def doc_text(self, t): return self.dp + t
    def query_text(self, t): return self.qp + t

    def _encode(self, texts, batch_size):
        self.calls += len(texts)
        X = np.zeros((len(texts), self.dim), "float32")
        for i, t in enumerate(texts):
            for w in t.lower().split():
                X[i, sum(map(ord, w)) % self.dim] += 1
        return self.bad(X) if self.bad else X


def mk_chunks():
    txt = ["transformers use attention for language modelling", "ride sharing matching algorithm for taxis",
           "convolutional networks for image classification", "attention mechanisms in vision transformers"]
    return [Chunk(f"d{i % 2}", f"T{i}", f"c{i}", 1, 1, "methodology" if i % 2 else "results", t, "u")
            for i, t in enumerate(txt)]


def test_prefix_conventions():
    e5, bge, sb = E5Embedder("x"), BGEEmbedder("x"), SBERTEmbedder("x")
    assert e5.query_text("q") == "query: q" and e5.doc_text("p") == "passage: p"
    assert bge.query_text("q").startswith("Represent this sentence for searching relevant passages: ")
    assert bge.doc_text("p") == "p" and sb.query_text("q") == "q" == sb.doc_text("q")
    with pytest.raises(ValueError):
        make_embedder("nope")


def test_version_changes_with_prefix():
    assert FakeEmbedder().version != FakeEmbedder(qp="query: ").version
    assert FakeEmbedder().key.replace("_", "").isalnum()


def test_embeddings_normalised_and_validated():
    X = FakeEmbedder().embed_documents(["a b c", "d e"])
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    with pytest.raises(EmbeddingError, match="non-finite"):
        FakeEmbedder(bad=lambda X: X * np.nan).embed_documents(["a"])
    with pytest.raises(EmbeddingError, match="dimension"):
        FakeEmbedder(bad=lambda X: X[:, :10]).embed_documents(["a"])
    assert FakeEmbedder().embed_documents([]).shape == (0, 64)


def test_cache_avoids_recompute_and_isolated_per_version(tmp_path):
    m, ch = EmbeddingManager(tmp_path), mk_chunks()
    e = FakeEmbedder()
    X1, n1, _ = m.embed_chunks(e, ch)
    assert n1 == 4 and e.calls == 4
    X2, n2, _ = m.embed_chunks(e, ch)
    assert n2 == 0 and e.calls == 4 and np.allclose(X1, X2)
    e2 = FakeEmbedder(dp="passage: ")                 # different version => separate cache dir
    _, n3, _ = m.embed_chunks(e2, ch)
    assert n3 == 4 and e2.key != e.key


def test_stale_cache_version_ignored(tmp_path):
    c = EmbeddingCache(tmp_path, "k", "v1"); c.put(["h"], np.ones((1, 4), "float32"))
    assert EmbeddingCache(tmp_path, "k", "v1").get(["h"])
    assert EmbeddingCache(tmp_path, "k", "v2").get(["h"]) == {}


def test_build_index_persist_and_reuse(tmp_path):
    m, ch, e = EmbeddingManager(tmp_path), mk_chunks(), FakeEmbedder()
    s, st = m.build_dense_index(e, ch, "cfg")
    assert s.size == 4 and not st.reused_index and st.index_bytes > 0
    assert all(c.embedding_model_version == e.version for c in ch)
    calls = e.calls
    s2, st2 = m.build_dense_index(e, ch, "cfg")
    assert st2.reused_index and e.calls == calls
    # new model version => separate index dir, nothing mixed
    e3 = FakeEmbedder(qp="q: ")
    _, st3 = m.build_dense_index(e3, ch, "cfg")
    assert st3.index_dir != st.index_dir


def test_dense_retrieval_and_filters(tmp_path):
    m, ch, e = EmbeddingManager(tmp_path), mk_chunks(), FakeEmbedder()
    s, _ = m.build_dense_index(e, ch, "cfg")
    r = DenseRetriever(e, s)
    hits = r.retrieve("attention transformers", 4)
    assert hits[0].chunk_id in {"c0", "c3"} and [h.rank for h in hits] == [1, 2, 3, 4]
    assert hits[0].score >= hits[-1].score
    meta = {c.chunk_id: {"document_id": c.document_id, "section": c.section} for c in ch}
    only_d1 = r.retrieve("attention transformers", 4, make_filter(meta, document_ids={"d1"}))
    assert {h.chunk_id for h in only_d1} == {"c1", "c3"}
    assert {h.chunk_id for h in r.retrieve("x", 4, make_filter(meta, sections={"results"}))} == {"c0", "c2"}


def test_mixing_models_rejected(tmp_path):
    m, ch = EmbeddingManager(tmp_path), mk_chunks()
    s, _ = m.build_dense_index(FakeEmbedder(), ch, "cfg")
    with pytest.raises(ValueError, match="mixing"):
        DenseRetriever(FakeEmbedder(qp="query: "), s)


def test_faiss_save_load_and_guards(tmp_path):
    s = FaissStore(4, "v")
    X = np.eye(4, dtype="float32")
    s.add(list("abcd"), X)
    with pytest.raises(ValueError): s.add(["a"], X[:1])
    with pytest.raises(ValueError): s.add(["z"], np.ones((1, 3), "float32"))
    s.save(tmp_path)
    s2 = FaissStore.load(tmp_path)
    assert s2.search(X[2], 1)[0].chunk_id == "c" and s2.ids() == list("abcd")
    assert FaissStore(4, "v").search(X[0], 3) == []


def test_tfidf_index(tmp_path):
    ch = mk_chunks()
    t = TfidfIndex().fit([c.chunk_id for c in ch], [c.text for c in ch])
    assert t.search("ride sharing taxis", 3)[0].chunk_id == "c1"
    assert t.search("zzzzqqq", 3) == []                 # no lexical overlap => no hits
    t.save(tmp_path); t2 = TfidfIndex.load(tmp_path)
    assert t2.search("image classification", 1)[0].chunk_id == "c2" and t.version == t2.version
    assert SparseRetriever(t).health()
    with pytest.raises(ValueError): TfidfIndex().fit([], [])


def test_health_check_failure():
    class Broken(FakeEmbedder):
        def _encode(self, texts, batch_size): raise RuntimeError("model gone")
    assert not Broken().health_check()


@pytest.mark.slow
def test_real_models_smoke():
    """Uses locally cached checkpoints; skipped when they cannot be loaded (offline)."""
    for fam in ("sbert", "e5", "bge"):
        e = make_embedder(fam, device="cpu")
        try:
            e.load()
        except EmbeddingError:
            pytest.skip(f"{fam} checkpoint unavailable")
        d = e.embed_documents(["Transformers use self-attention.", "Taxi ride sharing optimisation."])
        q = e.embed_queries(["What is self-attention?"])[0]
        assert d.shape == (2, e.dim) and (d @ q).argmax() == 0, fam
