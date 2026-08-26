"""LLM abstraction. Answering is grounded strictly in retrieved chunks.

A real provider (OpenAI/Anthropic/etc.) can implement LLMClient and be
swapped in without touching app/answer.py. The default is a dependency-free
MockLLM so the service runs end-to-end with no paid API key.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import re


class LLMClient(ABC):
    @abstractmethod
    def generate_answer(self, question: str, context_snippets: list[str]) -> str:
        """Given a question and a list of grounding snippets, return an answer
        composed only from those snippets (no outside knowledge)."""
        raise NotImplementedError

    def generate_answer_with_sources(
        self, question: str, context_snippets: list[str]
    ) -> tuple[str, list[int]]:
        """Optional richer variant returning which snippet indices were used.
        Default implementation falls back to attributing all snippets."""
        answer = self.generate_answer(question, context_snippets)
        used = list(range(len(context_snippets))) if answer else []
        return answer, used


class MockLLM(LLMClient):
    """Extractive stub: no external API, no invented facts.

    Relevance is already decided upstream by semantic (embedding) retrieval
    in app/answer.py — only chunks that cleared the similarity threshold are
    ever passed in here. This class's job is narrower: given context already
    judged relevant, pick and return the specific sentence(s) that best
    answer the question, verbatim, so every word in the answer is traceable
    to the retrieved text and nothing is invented.

    Sentences are ranked by lexical keyword overlap with the question. This
    is a ranking signal, not a relevance gate — a sentence doesn't need to
    literally repeat the question's words to be included, it just needs to
    come from a chunk that semantic retrieval already judged relevant. If no
    sentence has any keyword overlap at all, the first sentences of the
    retrieved chunks are used as a fallback (still 100% drawn from retrieved
    text, never generated).
    """

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "and", "or", "for", "what", "when", "where", "who", "how", "does",
        "do", "did", "which", "with", "this", "that", "it", "as", "be", "by",
        "from", "at", "your", "you", "can", "will", "should", "mentioned",
        "mentioned in",
    }

    def _keywords(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {w for w in words if w not in self._STOPWORDS and len(w) > 2}

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def generate_answer(self, question: str, context_snippets: list[str]) -> str:
        text, _ = self.generate_answer_with_sources(question, context_snippets)
        return text

    def generate_answer_with_sources(
        self, question: str, context_snippets: list[str]
    ) -> tuple[str, list[int]]:
        """Like generate_answer, but also returns the indices (into
        context_snippets) that were actually used, so callers can cite only
        what was drawn on rather than everything retrieved."""
        if not context_snippets:
            return "", []

        q_kw = self._keywords(question)
        scored: list[tuple[float, str, int]] = []
        for i, snippet in enumerate(context_snippets):
            for sent in self._split_sentences(snippet):
                s_kw = self._keywords(sent)
                overlap = len(q_kw & s_kw)
                score = overlap / (len(s_kw) ** 0.5) if s_kw else 0.0
                scored.append((score, sent, i))

        if not scored:
            return "", []

        # Sentences with lexical overlap are preferred; if none exist,
        # fall back to the top-ranked (i.e. most semantically relevant)
        # retrieved chunk's own sentences, in order.
        with_overlap = [s for s in scored if s[0] > 0]
        pool = with_overlap if with_overlap else scored[: max(3, len(scored))]
        pool.sort(key=lambda x: x[0], reverse=True)

        top_sentences: list[str] = []
        used_indices: list[int] = []
        seen: set[str] = set()
        for _, sent, idx in pool:
            if sent not in seen:
                top_sentences.append(sent)
                seen.add(sent)
                if idx not in used_indices:
                    used_indices.append(idx)
            if len(top_sentences) >= 6:
                break
        return " ".join(top_sentences), used_indices


def get_llm_client() -> LLMClient:
    return MockLLM()
