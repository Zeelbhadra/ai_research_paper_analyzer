"""Content-grounding tests: sentence splitting with citations, evidence highlighting, per-sentence evidence."""
from backend.explainability.evidence_explainer import evidence_strength, explain_answer, highlight_spans
from backend.preprocessing.chunking import Chunk
from backend.rag.citation_manager import analyze_answer, build_references, split_sentences
from backend.rag.context_builder import build_context

BUL = "•"


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


def make_response(ans, chunks):
    items = build_context("q", [(c, 1.0, i + 1) for i, c in enumerate(chunks)], per_doc_cap=None).items
    a = analyze_answer(ans, items)
    return {"answer": ans, "answer_mode": "llm", "verification": None, "citation_quality": {"unsupported_claims": a.unsupported},
            "references": build_references(items, a, "q"),
            "sentences": [{"text": s.text, "citations": s.citations, "invalid_citations": s.invalid_citations, "status": s.status,
                           "support": s.support, "supporting_label": s.supporting_label} for s in a.sentences]}


def test_explain_answer_distinguishes_stated_interpretation_and_unverified():
    chunks = [Chunk("A", "Paper A", "A:0", 3, 3, "methodology",
                    "The Swin Transformer computes attention inside shifted windows. It reaches 87.3 top-1 accuracy on ImageNet.", "http://a", None, 0),
              Chunk("B", "Paper B", "B:0", 7, 8, "results",
                    "Dual decomposition solvers give the tightest lower bounds on the benchmark datasets.", "http://b", None, 0)]
    ans = ("The Swin Transformer computes attention inside shifted windows [S1]. "
           "Dual decomposition solvers give the tightest lower bounds on the benchmark datasets [S2]. "
           "Interpretation: windowed attention is probably cheaper [S1]. "
           "Quantum annealing outperforms every classical solver by far [S2]. "
           "The weather is lovely in spring and summer.")
    e = explain_answer(make_response(ans, chunks))
    kinds = {x["text"][:20]: x["kind"] for x in e["sentences"]}
    assert kinds["The Swin Transformer"] == "stated_in_source" and kinds["Interpretation: wind"] == "interpretation"
    assert kinds["Quantum annealing ou"] == "unverified" and kinds["The weather is lovel"] == "unverified"
    first = e["sentences"][0]
    assert first["evidence"][0]["pages"] == [3, 3] and first["evidence"][0]["highlights"] and first["evidence"][0]["section"] == "methodology"
    assert e["papers"]["A"]["claims_cited"] == 1 and e["papers"]["B"]["claims_cited"] == 2
    assert next(x for x in e["sentences"] if x["text"].startswith("The weather"))["warning"].startswith("No valid citation")


def test_extractive_answer_note_warns_about_verbatim_strength():
    resp = {"answer": "x", "answer_mode": "extractive", "verification": None, "citation_quality": None, "references": [], "sentences": []}
    assert "provenance only" in explain_answer(resp)["note"]
