"""Retrieval + grounded answering + abstention logic."""
from __future__ import annotations
from pydantic import BaseModel

from app.store import DocumentIndex
from app.llm import get_llm_client

TOP_K = 5
# Cosine similarity from all-MiniLM-L6-v2 sentence embeddings. The absolute
# floor is deliberately conservative; a small score band around a strong best
# match lets fine-grained related chunks contribute without admitting weak,
# unrelated documents.
SIMILARITY_THRESHOLD = 0.28
RELATED_SCORE_BAND = 0.20


class Citation(BaseModel):
    page: int
    chunk_id: str


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


ABSTAIN_MESSAGE = (
    "The provided document does not contain information to answer this question."
)


def answer_question(index: DocumentIndex, question: str) -> AnswerResult:
    results = index.search(question, k=TOP_K)

    # Abstain if nothing retrieved, or best match is too weak to trust.
    if not results:
        return AnswerResult(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)

    best_score = results[0][1]
    same_page_support = any(
        chunk.page == results[0][0].page and score >= best_score - 0.05
        for chunk, score in results[1:]
    )
    if best_score < SIMILARITY_THRESHOLD and not (best_score >= 0.20 and same_page_support):
        return AnswerResult(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)

    # Keep only chunks that clear the relevance bar; these are the sole
    # grounding context passed to the LLM.
    candidate_floor = max(SIMILARITY_THRESHOLD - RELATED_SCORE_BAND, best_score - RELATED_SCORE_BAND)
    relevant = [(chunk, score) for chunk, score in results if score >= candidate_floor]
    if not relevant:
        return AnswerResult(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)

    context_snippets = [chunk.text for chunk, _ in relevant]
    llm = get_llm_client()
    answer_text, used_indices = llm.generate_answer_with_sources(question, context_snippets)

    if not answer_text.strip() or not used_indices:
        return AnswerResult(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)

    citations = [
        Citation(page=relevant[i][0].page, chunk_id=relevant[i][0].chunk_id)
        for i in used_indices
    ]
    return AnswerResult(answer=answer_text, citations=citations, abstained=False)
