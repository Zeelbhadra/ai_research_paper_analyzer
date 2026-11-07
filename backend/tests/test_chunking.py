import numpy as np
import pytest

from backend.preprocessing.chunking import ChunkConfig, chunk_document, _n
from backend.preprocessing.section_extractor import build_structured_text, classify_heading
from backend.preprocessing.text_cleaner import clean_pages, normalize, remove_headers_footers

PAGES = [
    "Running Head 1\nAbstract\nWe study things. It works well. More text here for the abstract part.\nPage 1",
    "Running Head 2\n1 Introduction\n" + "Intro sentence number one is here. " * 40 + "\n2",
    "Running Head 3\n2. Methodology\n" + "We apply a method to the data carefully. " * 40 + "\n3",
    "Running Head 4\nResults\n" + "The results show strong gains overall. " * 40 + "\n4",
    "Running Head 5\nReferences\n[1] Someone. Title. 2020.\n5",
]


def test_headings():
    assert classify_heading("3. Methodology") == "methodology"
    assert classify_heading("IV. EXPERIMENTS") == "results"
    assert classify_heading("Abstract") == "abstract"
    assert classify_heading("Abstract—We propose x") == "abstract"
    assert classify_heading("The method works because of reasons that are long enough to be a sentence.") is None


def test_normalize_dehyphenate_and_ligatures():
    assert normalize("compu-\ntation of ﬁnal") == "computation of final"


def test_header_footer_removal():
    out = remove_headers_footers(PAGES)
    assert all("Running Head" not in p for p in out)
    assert all(not p.strip().endswith(str(i + 1)) or len(p) > 100 for i, p in enumerate(out))
    # too few pages: no inference of repetition, but page numbers still go
    few = remove_headers_footers(["Head\nbody text\n1", "Head\nmore\n2"])
    assert "Head" in few[0] and not few[0].endswith("1")


@pytest.fixture
def st():
    return build_structured_text(clean_pages(PAGES))


def test_section_and_page_locate(st):
    assert st.section_names[:5] == ["abstract", "introduction", "methodology", "results", "references"]
    pos = st.text.index("We apply a method")
    assert st.page_at(pos) == 3 and st.section_at(pos) == "methodology"


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "sentence", "section"])
@pytest.mark.parametrize("size", [64, 128])
def test_strategies_respect_size_and_provenance(st, strategy, size):
    cfg = ChunkConfig(strategy, size, 0.2, min_chars=20)
    chunks = chunk_document(st, "doc1", "T", "http://u", cfg)
    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert _n(c.text) <= size * 1.05 + 2, (strategy, _n(c.text))
        assert c.document_id == "doc1" and c.paper_title == "T" and c.source_url == "http://u"
        assert 1 <= c.page_start <= c.page_end <= 5
        assert c.text in st.text
        assert c.section != "references"
        assert c.embedding_model_version is None


def test_section_aware_never_crosses_sections(st):
    for c in chunk_document(st, "d", "T", "", ChunkConfig("section", 4000, 0, min_chars=20)):
        a = st.text.index(c.text)
        assert len({st.section_at(a), st.section_at(a + len(c.text) - 1)}) == 1


def test_overlap_present_in_fixed(st):
    c = chunk_document(st, "d", "T", "", ChunkConfig("fixed", 50, 0.2, min_chars=20))
    assert c[0].text.split()[-5:] == c[1].text.split()[:5] or set(c[0].text.split()[-10:]) & set(c[1].text.split()[:10])


def test_semantic_requires_embedder_and_works(st):
    with pytest.raises(ValueError):
        chunk_document(st, "d", "T", "", ChunkConfig("semantic", 100))
    rng = np.random.default_rng(0)
    emb = lambda xs: rng.normal(size=(len(xs), 8))
    chunks = chunk_document(st, "d", "T", "", ChunkConfig("semantic", 100, 0.1, min_chars=20), embed=emb)
    assert chunks and all(_n(c.text) <= 110 for c in chunks)


def test_unknown_strategy(st):
    with pytest.raises(ValueError):
        chunk_document(st, "d", "T", "", ChunkConfig("bogus"))


def test_chunk_config_key():
    assert ChunkConfig("fixed", 512, 0.15).key() == "fixed-512-15"
