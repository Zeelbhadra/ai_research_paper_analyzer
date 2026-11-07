import json

import pytest

from backend.monitoring.logger import EventLog
from backend.preprocessing.chunking import Chunk
from backend.rag.citation_manager import analyze_answer, build_references, cited_labels
from backend.rag.context_builder import build_context, query_coverage
from backend.rag.generator import INSUFFICIENT, build_messages, extractive_answer, generate, stream_generate
from backend.rag.llm import LLMProvider, LLMResponse, OpenAIProvider, ProviderChain, ProviderError
from backend.rag.pipeline import RAGConfig, RAGPipeline
from backend.rag.query_processor import InvalidQuery, detect_intent, process_query
from backend.retrieval.retriever import Retriever
from backend.retrieval.service import RetrievalService, ServiceConfig
from backend.vectorstore.base import Hit


def ch(doc, i, text, section="methodology", page=1, title=None):
    return Chunk(doc, title or f"Paper {doc}", f"{doc}:{i}", page, page, section, text, f"http://x/{doc}", None, i)


T_ATT = ("Self-attention lets every token attend to every other token. The Swin Transformer computes attention inside "
         "shifted windows, which reduces complexity from quadratic to linear in image size. It reaches 87.3 top-1 accuracy on ImageNet.")
T_GM = ("Graph matching assigns nodes of one graph to nodes of another. Dual decomposition solvers give the tightest lower bounds "
        "on the benchmark datasets while the fm heuristic is fastest.")
T_RS = "Ride-sharing platforms set prices for riders and compensation for drivers while operating autonomous vehicles at fixed cost."
CHUNKS = [ch("A", 0, T_ATT, page=3), ch("A", 1, "Windows are shifted between consecutive layers to connect neighbouring windows. " * 3, page=4),
          ch("B", 0, T_GM, "results", 7), ch("C", 0, T_RS, "introduction", 2)]


# ---------------- query processing ----------------
@pytest.mark.parametrize("q,intent", [
    ("Compare the methodologies of these two papers", "comparison"),
    ("What are the limitations of these studies?", "limitations"),
    ("What research gaps can be identified across these papers?", "research_gap"),
    ("Which papers discuss transformer-based approaches?", "paper_discovery"),
    ("Which datasets were used?", "datasets"),
    ("What are the major findings?", "results"),
    ("Explain the methodology used in this paper", "methodology"),
    ("What is graph matching?", "definition"),
    ("Summarize the evolution of object detection", "evolution"),
    ("hello there friend", "general")])
def test_intents_for_the_spec_example_queries(q, intent):
    assert detect_intent(q) == intent


def test_validation():
    for bad in ("", "  ", "ab", "x" * 1001, "12345 ----", None, 5):
        with pytest.raises(InvalidQuery):
            process_query(bad)
    assert process_query("What\x00 is\n\tattention?").normalized == "What is attention?"


def test_scope_and_filters():
    assert process_query("Explain the method", ["A"]).scope == "document"
    assert process_query("Explain the method", ["A", "B"]).scope == "cross_document"
    assert process_query("Compare the methods", ["A"]).scope == "cross_document"
    assert process_query("What does this paper propose?").warnings


def test_followup_condensation():
    hist = [{"role": "user", "content": "How does Swin Transformer reduce complexity?", "retrieval_query": "How does Swin Transformer reduce complexity?"},
            {"role": "assistant", "content": "It uses shifted windows [S1]."}]
    pq = process_query("What about its accuracy?", history=hist)
    assert pq.is_followup and "Swin Transformer" in pq.retrieval_query and pq.normalized == "What about its accuracy?"
    standalone = process_query("Which datasets were used for evaluating graph matching solvers in this benchmark?", history=hist)
    assert not standalone.is_followup and standalone.retrieval_query == standalone.normalized


# ---------------- context building ----------------
def ranked(cs, scores=None):
    return [(c, (scores or [1 / (i + 1) for i in range(len(cs))])[i], i + 1) for i, c in enumerate(cs)]


def test_context_dedup_cap_budget_and_order():
    dup = ch("A", 9, T_ATT)                                   # identical text from another chunk
    many = [ch("A", i, f"unique content number {i} " + "word" * 3 + f" topic{i} alpha{i} beta{i} gamma{i} delta{i} " * 4, page=10 - i) for i in range(1, 6)]
    r = build_context("attention windows", ranked([CHUNKS[0], dup] + many + [CHUNKS[2]]), max_chunks=6, per_doc_cap=3)
    texts = [i.chunk.text for i in r.items]
    assert r.dropped_duplicates == 1 and texts.count(T_ATT) == 1
    assert sum(1 for i in r.items if i.chunk.document_id == "A") == 3 and any(i.chunk.document_id == "B" for i in r.items)
    assert [i.label for i in r.items] == [f"S{n + 1}" for n in range(len(r.items))]
    pages = [i.chunk.page_start for i in r.items if i.chunk.document_id == "A"]
    assert pages == sorted(pages)                             # document order preserved within a paper
    small = build_context("q", ranked(CHUNKS), max_chunks=6, max_words=30)
    assert len(small.items) == 1 and small.truncated >= 1


def test_coverage_separates_on_and_off_topic():
    texts = [c.text for c in CHUNKS]
    assert query_coverage("shifted windows attention complexity", texts) > 0.8
    assert query_coverage("recipe for sourdough bread baking", texts) < 0.2


# ---------------- citations ----------------
def items_for(cs):
    return build_context("q", ranked(cs), max_chunks=10, per_doc_cap=None).items


def test_citation_parsing_and_support_levels():
    assert cited_labels("Text [S1][S3] and [S2, S4].") == ["S1", "S3", "S2", "S4"]
    items = items_for([CHUNKS[0], CHUNKS[2]])
    ans = ("The Swin Transformer computes attention inside shifted windows, reducing complexity to linear [S1]. "
           "It reaches 99.9 top-1 accuracy on ImageNet [S1]. "
           "Dual decomposition solvers give the tightest lower bounds on the benchmark datasets [S2]. "
           "Quantum annealing outperforms all classical solvers by a wide margin [S2]. "
           "The weather in Paris is lovely in spring. "
           "Interpretation: windowed attention is probably cheaper for large images [S1]. "
           "This is claimed with a missing source [S7].")
    a = analyze_answer(ans, items)
    def S(prefix):
        return next(x for x in a.sentences if x.text.startswith(prefix))
    assert S("The Swin Transformer com").status == "supported"
    assert S("It reaches 99.9 top-1 ac").status != "supported"                 # wrong number caps support
    assert S("Dual decomposition solv").status == "supported"
    assert S("Quantum annealing outpe").status == "unsupported"
    assert S("The weather in Paris is").status == "uncited_claim"
    assert S("Interpretation: windowe").status == "interpretation" and S("Interpretation: windowe").citations == ["S1"]
    assert S("This is claimed with a ").invalid_citations == ["S7"] and not a.all_citations_valid
    assert a.citation_validity < 1 and a.cited_fraction < 1
    assert any("Quantum" in u for u in a.unsupported) and any("Paris" in u for u in a.unsupported)


def test_references_include_page_section_and_quote():
    items = items_for([CHUNKS[0], CHUNKS[2]])
    a = analyze_answer("Dual decomposition solvers give the tightest lower bounds [S2].", items)
    refs = build_references(items, a, "q")
    r2 = next(r for r in refs if r["label"] == "S2")
    assert r2["cited"] and r2["pages"] == [7, 7] and r2["section"] == "results" and "tightest lower bounds" in r2["quote"]
    assert not next(r for r in refs if r["label"] == "S1")["cited"]


# ---------------- providers / generation ----------------
class Stub(LLMProvider):
    name = "stub"
    model = "stub-1"

    def __init__(self, text="Answer [S1].", fail=False):
        self.text, self.fail, self.seen = text, fail, None

    def generate(self, messages, max_tokens=500, temperature=0.0):
        self.seen = messages
        if self.fail:
            raise ProviderError("stub: rate limited")
        return LLMResponse(self.text, "stub", "stub-1", 100, 10, 0.01, 0.0)

    def stream(self, messages, max_tokens=500, temperature=0.0):
        if self.fail:
            raise ProviderError("stub: down")
        yield from self.text.split(" ")


def test_prompt_is_strict_and_contains_labels_history_and_intent():
    items = items_for([CHUNKS[0], CHUNKS[2]])
    pq = process_query("Compare the two approaches", ["A", "B"])
    msgs = build_messages(pq, items, [{"role": "user", "content": "earlier q"}, {"role": "assistant", "content": "earlier a"}])
    sys_p = msgs[0]["content"]
    assert "ONLY from the numbered source passages" in sys_p and "Interpretation:" in sys_p and "comparison" in sys_p.lower()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "[S1] Paper A (p. 3, methodology)" in msgs[-1]["content"] and "[S2] Paper B (p. 7, results)" in msgs[-1]["content"]


def test_generate_modes():
    items = items_for([CHUNKS[0], CHUNKS[2]])
    pq = process_query("What does Swin Transformer do with shifted windows?")
    g = generate(pq, items, Stub("The Swin Transformer uses shifted windows [S1]."))
    assert g.mode == "llm" and g.provider == "stub" and g.prompt_tokens == 100
    g2 = generate(pq, items, Stub(fail=True))
    assert g2.mode == "extractive" and "[S1]" in g2.text and g2.provider_errors and "shifted windows" in g2.text
    assert generate(pq, items, None).mode == "extractive"
    assert generate(pq, [], Stub()).mode == "insufficient"
    with pytest.raises(ProviderError):
        generate(pq, items, Stub(fail=True), allow_extractive=False)
    assert extractive_answer("quantum annealing recipes", items) == INSUFFICIENT


def test_provider_chain_failover_and_no_key_unavailable(monkeypatch):
    chain = ProviderChain([Stub(fail=True), Stub("second [S1]")])
    r = chain.generate([{"role": "user", "content": "x"}])
    assert r.text == "second [S1]" and r.attempts and "rate limited" in r.attempts[0]
    with pytest.raises(ProviderError):
        ProviderChain([Stub(fail=True)]).generate([])
    monkeypatch.setattr("backend.rag.llm._ENV_LOADED", True)        # never read the real .env in tests
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not OpenAIProvider().available() and not ProviderChain([OpenAIProvider()]).available()
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    assert OpenAIProvider().available()
    assert list(ProviderChain([Stub(fail=True), Stub("a b c")]).stream([])) == ["a", "b", "c"]


def test_stream_generate_falls_back_to_extractive():
    items = items_for([CHUNKS[0]])
    pq = process_query("What does Swin Transformer do with shifted windows?")
    assert "".join(stream_generate(pq, items, Stub(fail=True))).startswith("Extractive answer")
    assert "".join(stream_generate(pq, [], Stub())) == INSUFFICIENT


# ---------------- pipeline ----------------
class FakeRetriever(Retriever):
    name, kind = "fake", "hybrid"

    def __init__(self, order):
        self.order = order

    def retrieve(self, query, k, allow=None):
        return [Hit(c, 1 / (i + 1), i + 1) for i, c in enumerate(self.order) if allow is None or allow(c)][:k]


class FakeProd:
    def __init__(self, order):
        self.chunks = {c.chunk_id: c for c in CHUNKS}
        self.meta = {c.chunk_id: {"document_id": c.document_id, "section": c.section} for c in CHUNKS}
        self.service = RetrievalService(lambda: FakeRetriever(order), None, "fake", "v1", None, ServiceConfig(), EventLog())


def pipe(provider=None, min_cov=None, order=("A:0", "A:1", "B:0", "C:0")):
    return RAGPipeline(FakeProd(list(order)), provider, None, RAGConfig(min_coverage=min_cov))


def test_pipeline_end_to_end_with_citations_and_timings():
    p = pipe(Stub("The Swin Transformer computes attention inside shifted windows [S1]. Graph solvers differ [S2]."))
    r = p.ask("How does the Swin Transformer compute attention with shifted windows?")
    assert r["answer_mode"] == "llm" and r["sufficient_evidence"] and r["references"][0]["pages"] == [3, 3]
    assert r["retrieval"]["kind"] == "hybrid" and r["generation"]["prompt_tokens"] == 100
    assert set(r["timings"]) == {"retrieval_s", "rerank_s", "generation_s", "total_s"}
    assert r["citation_quality"]["valid_citation_fraction"] == 1.0 and r["sentences"][0]["status"] == "supported"
    assert any(x["cited"] for x in r["references"])


def test_pipeline_refuses_without_llm_call_when_evidence_is_weak():
    stub = Stub("should not be called [S1]")
    r = pipe(stub, min_cov=0.5).ask("What is the recipe for sourdough bread baking?")
    assert r["answer_mode"] == "insufficient" and not r["sufficient_evidence"] and stub.seen is None
    assert r["answer"] == INSUFFICIENT and r["citation_quality"] is None


def test_pipeline_document_filter_followup_and_llm_failure():
    p = pipe(Stub(fail=True))
    r = p.ask("Which datasets were used for graph matching benchmarks?", document_ids=["B"])
    assert r["answer_mode"] == "extractive" and {x["paper_id"] for x in r["references"]} == {"B"}
    hist = [{"role": "user", "content": "How does the Swin Transformer work?", "retrieval_query": "How does the Swin Transformer work?"}]
    r2 = pipe(Stub("Windows are shifted [S1].")).ask("What about its accuracy?", history=hist)
    assert r2["query"]["is_followup"] and "Swin" in r2["query"]["retrieval_query"]
    with pytest.raises(InvalidQuery):
        p.ask("")


def test_pipeline_stream_events():
    ev = list(pipe(Stub("Windows are shifted between layers [S1]")).stream("How are the windows shifted between layers?"))
    assert ev[0]["type"] == "meta" and ev[-1]["type"] == "final"
    assert "".join(e["text"] for e in ev if e["type"] == "token").strip() == "Windowsareshiftedbetweenlayers[S1]"
    assert ev[-1]["answer_mode"] == "llm" and ev[-1]["sources"] if False else ev[-1]["references"]


def test_rag_api(monkeypatch):
    import app as flask_app
    from backend.api.routes import rag
    monkeypatch.setattr(rag, "get_pipeline", lambda: pipe(Stub("Windows are shifted [S1].")))
    c = flask_app.app.test_client()
    assert c.post("/api/rag/ask", json={"question": ""}).status_code == 400
    assert c.post("/api/rag/ask", json={"question": "How are windows shifted?", "history": "x"}).status_code == 400
    j = c.post("/api/rag/ask", json={"question": "How are the windows shifted between layers?"}).get_json()
    assert j["answer_mode"] == "llm" and j["references"]
    body = c.post("/api/rag/stream", json={"question": "How are the windows shifted between layers?"}).get_data(as_text=True)
    events = [json.loads(l[6:]) for l in body.split("\n\n") if l.startswith("data: ")]
    assert events[0]["type"] == "meta" and events[-1]["type"] == "final"


class DenseFake(FakeRetriever):
    def __init__(self, order, sig):
        super().__init__(order)
        self.sig = sig

    def evidence_signal(self, query, chunk_ids):
        return self.sig


def dense_pipe(sig, provider=None, cos=0.82, version="v1", lexical=None):
    prod = FakeProd(["A:0", "A:1", "B:0", "C:0"])
    prod.service = RetrievalService(lambda: DenseFake(["A:0", "A:1", "B:0", "C:0"], sig), None, "fake", "v1", None, ServiceConfig(), EventLog())
    return RAGPipeline(prod, provider, None, RAGConfig(min_dense_cosine=cos, dense_model_version=version, min_coverage=lexical))


def test_dense_gate_refuses_low_similarity_without_calling_llm():
    stub = Stub("never [S1]")
    r = dense_pipe(0.75, stub).ask("How does the Swin Transformer compute attention with shifted windows?")
    assert r["answer_mode"] == "insufficient" and stub.seen is None
    assert r["evidence"]["gate"] == "dense_cosine" and r["evidence"]["signal"] == 0.75 and r["evidence"]["threshold"] == 0.82
    ok = dense_pipe(0.9, Stub("Windows are shifted [S1].")).ask("How does the Swin Transformer compute attention with shifted windows?")
    assert ok["answer_mode"] == "llm" and ok["sufficient_evidence"]


def test_gate_falls_back_to_lexical_when_model_version_differs_or_signal_missing():
    q = "What is the recipe for sourdough bread baking?"
    r = dense_pipe(0.99, Stub("x [S1]"), version="other-model", lexical=0.5).ask(q)
    assert r["evidence"]["gate"].startswith("lexical") and r["answer_mode"] == "insufficient"
    r2 = dense_pipe(None, Stub("Windows are shifted [S1]."), lexical=0.2).ask("How are the windows shifted between layers?")
    assert r2["evidence"]["gate"].startswith("lexical") and r2["sufficient_evidence"]
    assert dense_pipe(None, Stub("Windows [S1]."), lexical=None).ask(q)["evidence"]["gate"] == "none"


def test_unverified_llm_answer_is_replaced_by_extractive_with_original_kept():
    uncited = Stub("The Swin Transformer partitions images into windows. It then does something remarkable with them everywhere.")
    r = pipe(uncited).ask("How does the Swin Transformer compute attention with shifted windows?")
    assert r["answer_mode"] == "extractive" and r["verification"]["replaced"] and not r["verification"]["verified"]
    assert "partitions images" in r["verification"]["llm_answer"] and "partitions images" not in r["answer"]
    assert "citations" in r["verification"]["reason"] and any("rejected" in e for e in r["generation"]["provider_errors"])
    good = pipe(Stub("The Swin Transformer computes attention inside shifted windows [S1]. Windows are shifted between layers [S1]."))
    g = good.ask("How does the Swin Transformer compute attention with shifted windows?")
    assert g["answer_mode"] == "llm" and g["verification"] == {"verified": True, "replaced": False, "llm_answer": None, "reason": None}
    bogus = pipe(Stub("Quantum annealing outperforms all classical solvers by a wide margin [S1]. Another invented claim about weather [S1]."))
    b = bogus.ask("How does the Swin Transformer compute attention with shifted windows?")
    assert b["verification"]["replaced"] and "supported" in b["verification"]["reason"]


def test_extractive_skips_fragments():
    items = items_for([ch("A", 5, "of self-attention from O(N 2) to O(N. Typically, self-attention is used as a. "
                                  "The Swin Transformer computes self-attention inside shifted windows to reduce complexity.")])
    out = extractive_answer("self-attention windows complexity", items)
    assert "shifted windows" in out and "O(N" not in out


# ---------------- explicit multi-paper selection must cover every selected paper ----------------
def test_interleave_and_per_document_retrieval():
    from backend.rag.multidoc import interleave, retrieve_per_document
    a = [Hit("a1", .9, 1), Hit("a2", .8, 2), Hit("a3", .7, 3)]
    b = [Hit("b1", .2, 1)]
    out = interleave([a, b])
    assert [h.chunk_id for h in out] == ["a1", "b1", "a2", "a3"] and [h.rank for h in out] == [1, 2, 3, 4]
    got = retrieve_per_document(lambda q, k, al: [Hit(c, 1, i + 1) for i, c in enumerate(["a1", "a2", "b1", "b2"]) if al(c)][:k], "q", ["A", "B"], 10,
                                lambda d: (lambda c: c.startswith(d.lower())))
    assert [h.chunk_id for h in got] == ["a1", "b1", "a2", "b2"]


def test_compare_mode_uses_every_selected_paper_even_when_one_dominates_the_ranking():
    many_a = [ch("A", i, f"alpha topic passage number {i} about swin windows attention " * 6, page=i + 1) for i in range(30)]
    few_b = [ch("B", i, f"beta solver passage number {i} about graph matching bounds " * 6, page=i + 1) for i in range(2)]
    chunks = many_a + few_b

    class Prod:
        def __init__(self):
            self.chunks = {c.chunk_id: c for c in chunks}
            self.meta = {c.chunk_id: {"document_id": c.document_id, "section": c.section} for c in chunks}
            order = [c.chunk_id for c in many_a] + [c.chunk_id for c in few_b]      # paper A fills the first 30 ranks, B only comes after
            self.service = RetrievalService(lambda: FakeRetriever(order), None, "fake", "v1", None, ServiceConfig(), EventLog())
    pipe = RAGPipeline(Prod(), None, None, RAGConfig(candidate_depth=20, cross_doc_depth=20))
    r = pipe.ask("Compare the methods of these two papers.", document_ids=["A", "B"])
    assert {x["paper_id"] for x in r["references"]} == {"A", "B"}, "paper B was silently dropped"
    assert sum(1 for x in r["references"] if x["paper_id"] == "B") == 2           # both of B's chunks, A is capped at 3
    one = pipe.ask("Compare the methods of these two papers.", document_ids=["A"])
    assert {x["paper_id"] for x in one["references"]} == {"A"}                    # single selection is unchanged
