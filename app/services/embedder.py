"""
Sentence embedding service using sentence-transformers (all-MiniLM-L6-v2).

Runs on CPU alongside the main GPU model.
Singleton — loaded once at startup, reused for every request.

Usage:
    from app.services.embedder import get_embedder
    emb = get_embedder()
    vector = emb.encode("How does bubble sort work?")  # np.ndarray [384]
    vectors = emb.encode_batch(["turn 1 text", "turn 2 text"])  # np.ndarray [N, 384]
"""

import numpy as np
from typing import List, Union

_embedder = None


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string → np.ndarray [embedding_dim]."""
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings → np.ndarray [N, embedding_dim]."""
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors (dot product suffices)."""
        # Both vectors are already L2-normalised by encode(), so cosine = dot product
        return float(np.dot(a, b))


def get_embedder(model_name: str = None) -> Embedder:
    """Return the singleton Embedder, creating it on first call."""
    global _embedder
    if _embedder is None:
        from app.config import EMBEDDER_MODEL
        _embedder = Embedder(model_name or EMBEDDER_MODEL)
    return _embedder
