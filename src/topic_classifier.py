"""
Topic Classifier
Classifies research papers into academic topics using zero-shot classification.
"""

from transformers import pipeline

CANDIDATE_LABELS = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Biology",
    "Medicine",
    "Economics",
    "Engineering",
    "Chemistry",
    "Environmental Science",
    "Psychology",
]

# Lazy-loaded singleton
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
    return _classifier


def classify_topic(
    text: str,
    candidate_labels: list[str] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Classify text into research topics using zero-shot classification.

    Args:
        text: Input text (abstract or full paper).
        candidate_labels: Custom label list; defaults to CANDIDATE_LABELS.
        top_n: Number of top predictions to return.

    Returns:
        List of dicts with 'label' and 'score', sorted by confidence.
    """
    if not text or len(text.strip()) < 20:
        return []

    labels = candidate_labels or CANDIDATE_LABELS
    classifier = _get_classifier()

    # Truncate to avoid OOM — zero-shot works fine on abstracts
    truncated = text[:2000]

    result = classifier(truncated, labels, multi_label=True)

    topics = []
    for label, score in zip(result["labels"], result["scores"]):
        topics.append({"label": label, "score": round(score, 4)})

    return topics[:top_n]
