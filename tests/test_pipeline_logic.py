"""Unit tests for the model-independent parts of the pipeline: the extractive
MockLLM and the retrieval-gate/abstention logic in answer.py. These use
synthetic chunks/scores directly (no embedding model involved), so they run
in any environment, including one without access to huggingface.co.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

from app.llm import MockLLM
from app.ingest import Chunk
from app.answer import answer_question, SIMILARITY_THRESHOLD


def test_mockllm_extracts_relevant_sentence_verbatim():
    llm = MockLLM()
    answer, used = llm.generate_answer_with_sources(
        "Where is OneRx headquartered?",
        ["OneRx was founded in 2019 and is headquartered in Boston, Massachusetts."],
    )
    assert "Boston" in answer
    assert used == [0]


def test_mockllm_answers_semantic_question_without_exact_keywords():
    """The regression case at the MockLLM layer: the question shares no
    exact vocabulary with the source sentence, but the source chunk has
    already been judged relevant upstream (that's the embedding retrieval's
    job) — MockLLM should still surface it rather than returning nothing."""
    llm = MockLLM()
    context = [
        "Built retrieval-augmented generation systems using large language "
        "models and vector search. Comfortable with Python and transformer-"
        "based embedding models."
    ]
    answer, used = llm.generate_answer_with_sources(
        "What AI technologies are mentioned?", context
    )
    assert answer.strip() != ""
    assert used == [0]


def test_mockllm_empty_context_returns_empty():
    llm = MockLLM()
    answer, used = llm.generate_answer_with_sources("Anything?", [])
    assert answer == ""
    assert used == []


def _chunk(page: int, chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", page=page, text=text)


def test_answer_question_abstains_below_threshold():
    """If nothing clears the similarity threshold, answer_question must
    abstain before ever calling the LLM layer."""
    fake_index = MagicMock()
    fake_index.search.return_value = [
        (_chunk(1, "c-0000", "Unrelated text."), SIMILARITY_THRESHOLD - 0.05),
    ]
    result = answer_question(fake_index, "irrelevant question")
    assert result.abstained is True
    assert result.citations == []


def test_answer_question_grounds_and_cites_above_threshold():
    fake_index = MagicMock()
    fake_index.search.return_value = [
        (_chunk(1, "c-0000", "OneRx is headquartered in Boston."), SIMILARITY_THRESHOLD + 0.3),
    ]
    result = answer_question(fake_index, "Where is OneRx headquartered?")
    assert result.abstained is False
    assert result.citations == [{"page": 1, "chunk_id": "c-0000"}] or (
        result.citations[0].page == 1 and result.citations[0].chunk_id == "c-0000"
    )
    assert "Boston" in result.answer


def test_answer_question_no_results_abstains():
    fake_index = MagicMock()
    fake_index.search.return_value = []
    result = answer_question(fake_index, "anything")
    assert result.abstained is True
    assert result.citations == []
