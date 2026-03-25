"""
Keyword Extractor
Extracts key terms from text using KeyBERT with a TF-IDF fallback.
"""

from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

# Lazy-loaded singleton
_kw_model = None


def _get_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _kw_model


def extract_keywords(
    text: str, top_n: int = 10, use_mmr: bool = True
) -> list[tuple[str, float]]:
    """
    Extract keywords from text using KeyBERT.

    Args:
        text: Input text.
        top_n: Number of keywords to return.
        use_mmr: Use Maximal Marginal Relevance for diversity.

    Returns:
        List of (keyword, score) tuples sorted by relevance.
    """
    if not text or len(text.strip()) < 20:
        return []

    try:
        model = _get_model()
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=0.5,
        )
        return keywords
    except Exception:
        return _tfidf_fallback(text, top_n)


def _tfidf_fallback(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Fallback keyword extraction using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        keywords = sorted(
            zip(feature_names, scores), key=lambda x: x[1], reverse=True
        )
        return keywords[:top_n]
    except Exception:
        return []
