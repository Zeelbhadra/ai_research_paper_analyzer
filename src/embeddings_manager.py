"""
Embeddings Manager
Creates sentence embeddings and performs similarity search using
Sentence-Transformers and FAISS.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

# Lazy-loaded singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class PaperIndex:
    """In-memory FAISS index for paper similarity search."""

    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.papers: list[dict] = []  # metadata store, aligned with index

    def add_paper(self, paper_meta: dict, text: str) -> int:
        """
        Encode text and add to the index.

        Args:
            paper_meta: Dict with at least 'title' key.
            text: Text to encode (abstract or full body).

        Returns:
            Index position of the added paper.
        """
        model = _get_model()
        embedding = model.encode([text], normalize_embeddings=True)
        embedding = np.array(embedding, dtype="float32")
        self.index.add(embedding)
        self.papers.append(paper_meta)
        return len(self.papers) - 1

    def find_similar(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Find the most similar papers to the query text.

        Returns:
            List of dicts with 'paper' metadata and 'distance'.
        """
        if self.index.ntotal == 0:
            return []

        model = _get_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype="float32")

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.papers):
                results.append({
                    "paper": self.papers[idx],
                    "distance": float(dist),
                })
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, directory: str):
        """Persist index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "papers.index"))
        with open(os.path.join(directory, "papers_meta.json"), "w") as f:
            json.dump(self.papers, f)

    def load(self, directory: str):
        """Load index and metadata from disk."""
        index_path = os.path.join(directory, "papers.index")
        meta_path = os.path.join(directory, "papers_meta.json")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path) as f:
                self.papers = json.load(f)


# Global shared index
paper_index = PaperIndex()
