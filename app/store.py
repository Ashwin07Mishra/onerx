"""In-memory per-document vector store.

Embeddings: a local Sentence Transformers model (default
`all-MiniLM-L6-v2`), run entirely on-device — no paid API key, no per-query
network call. The model is downloaded once from the Hugging Face Hub on
first use and then cached locally (~90MB), after which the service runs
fully offline. This replaces an earlier TF-IDF prototype: TF-IDF only
matches on shared vocabulary, so a question like "What AI technologies are
mentioned?" fails to retrieve a chunk that says "built RAG pipelines using
LangChain" (no literal word overlap) even though it's exactly the right
chunk. A sentence embedding model captures that semantic relationship.

Vector store: FAISS `IndexFlatIP` (unchanged) — one index per document,
built at ingest time, held in memory for the process lifetime. Chunk
embeddings are L2-normalized so inner product = cosine similarity.
"""
from __future__ import annotations
import threading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.ingest import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model (one instance per process,
    shared across all documents)."""
    global _model
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer(MODEL_NAME)
        return _model


class DocumentIndex:
    def __init__(self, doc_id: str, chunks: list[Chunk]):
        self.doc_id = doc_id
        self.chunks = chunks
        model = get_model()
        texts = [c.text for c in chunks]

        if texts:
            embeddings = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
        else:
            embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

        dim = embeddings.shape[1] if embeddings.shape[0] else model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)
        if len(embeddings):
            self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Returns up to k (Chunk, cosine_similarity) pairs, best first.
        Each Chunk carries doc_id, page, chunk_id, and text — the metadata
        required for citations is preserved end to end."""
        if not self.chunks:
            return []
        model = get_model()
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        k = min(k, len(self.chunks))
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


class Registry:
    """Holds all ingested documents by doc_id, in memory, process lifetime."""

    def __init__(self):
        self._docs: dict[str, DocumentIndex] = {}
        self._lock = threading.Lock()

    def add(self, doc_id: str, chunks: list[Chunk]) -> DocumentIndex:
        idx = DocumentIndex(doc_id, chunks)
        with self._lock:
            self._docs[doc_id] = idx
        return idx

    def get(self, doc_id: str) -> DocumentIndex | None:
        with self._lock:
            return self._docs.get(doc_id)


registry = Registry()
