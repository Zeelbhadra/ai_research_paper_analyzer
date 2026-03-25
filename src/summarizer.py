"""
Summarizer
Generates abstractive summaries using HuggingFace.
Falls back to extractive summaries using TF-IDF sentence scoring if the model fails.
"""

import re
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

# Lazy-loaded HuggingFace pipeline
_summarizer_pipeline = None


def _get_summarizer_pipeline():
    global _summarizer_pipeline

    if _summarizer_pipeline is None:
        try:
            from transformers import pipeline

            print("[Summarizer] Loading HuggingFace summarization model...")

            # UPDATED MODEL (more accurate for research papers)
            _summarizer_pipeline = pipeline(
                "summarization",
                model="google/pegasus-arxiv"
            )

            print("[Summarizer] HuggingFace model loaded successfully.")

        except Exception as e:

            print(f"[Summarizer] Failed to load HuggingFace model: {e}")

            _summarizer_pipeline = "FAILED"

    return _summarizer_pipeline


def _split_sentences(text: str) -> list[str]:

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _generate_tfidf_summary(text: str) -> str:

    sentences = _split_sentences(text)

    if len(sentences) <= 3:
        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(stop_words='english')

        tfidf_matrix = vectorizer.fit_transform(sentences)

        sentence_scores = tfidf_matrix.sum(axis=1).A1

        num_sentences = max(2, min(5, len(sentences) // 4))

        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

        top_indices.sort()

        summary_sentences = [sentences[i] for i in top_indices]

        return " ".join(summary_sentences)

    except Exception as e:

        print(f"[Summarizer fallback error] {e}")

        return " ".join(sentences[:3])


def generate_summary(text: str, max_length: int = 200, min_length: int = 60):

    if not text or len(text.strip()) < 100:
        return text.strip()

    summarizer = _get_summarizer_pipeline()

    if summarizer and summarizer != "FAILED":

        try:

            # Split long paper into chunks 
            chunk_size = 2000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            summaries = []

            for chunk in chunks[:5]:  # limit chunks to avoid very long runtime

                result = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

                summaries.append(result[0]["summary_text"])

            final_summary = " ".join(summaries)

            return final_summary

        except Exception as e:

            print(f"[Summarizer] HuggingFace generation failed: {e}")

            traceback.print_exc()

            print("[Summarizer] Falling back to TF-IDF summarization.")

    return _generate_tfidf_summary(text)