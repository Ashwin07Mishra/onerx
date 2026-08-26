import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import MODEL_NAME, get_model
from tests.make_test_pdf import build_test_pdf, build_resume_test_pdf

client = TestClient(app)


def _model_is_reachable() -> bool:
    """True if the sentence-transformers model can actually be loaded here
    (i.e. huggingface.co is reachable, or the model is already cached
    locally). False in network-restricted sandboxes where the model
    cannot be downloaded, in which case the model-dependent tests below
    are skipped rather than faked."""
    try:
        get_model()
        return True
    except Exception:
        return False


MODEL_AVAILABLE = _model_is_reachable()
_skip_reason = (
    f"'{MODEL_NAME}' could not be loaded in this environment "
    "(no network access to huggingface.co and no local cache). "
    "These tests require the real embedding model and are not faked."
)


requires_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason=_skip_reason)


@pytest.fixture(scope="module")
def doc_id():
    """OneRx company fixture: specific-fact questions (headquarters, launch
    year, CEO name) and the unanswerable/abstention case."""
    if not MODEL_AVAILABLE:
        pytest.skip(_skip_reason)
    pdf_bytes = build_test_pdf()
    resp = client.post(
        "/ingest",
        files={"file": ("fixture.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pages"] == 3
    assert data["chunks"] > 0
    return data["doc_id"]


@pytest.fixture(scope="module")
def resume_doc_id():
    """Resume-style fixture: used for the broader/semantic question case,
    where the right answer shares little literal vocabulary with the
    question ('AI technologies' vs. 'RAG systems ... large language
    models ... embedding models')."""
    if not MODEL_AVAILABLE:
        pytest.skip(_skip_reason)
    pdf_bytes = build_resume_test_pdf()
    resp = client.post(
        "/ingest",
        files={"file": ("resume_fixture.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pages"] == 3
    assert data["chunks"] > 0
    return data["doc_id"]


@requires_model
def test_ingest_returns_handle(doc_id):
    assert doc_id is not None


# --- Specific, exact-keyword-overlap questions -----------------------------

@requires_model
def test_answer_specific_fact_with_citation(doc_id):
    resp = client.post("/answer", json={"doc_id": doc_id, "question": "Where is OneRx headquartered?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert "Boston" in data["answer"]
    assert len(data["citations"]) > 0
    assert data["citations"][0]["page"] == 1
    assert "chunk_id" in data["citations"][0]


@requires_model
def test_answer_product_launch_year(doc_id):
    resp = client.post("/answer", json={"doc_id": doc_id, "question": "When was RxConnect launched?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert "2021" in data["answer"]
    assert any(c["page"] == 2 for c in data["citations"])


@requires_model
def test_answer_full_name_question(resume_doc_id):
    """Mirrors the reported case: a very specific, mostly-exact-keyword
    question should still be answered correctly."""
    resp = client.post(
        "/answer", json={"doc_id": resume_doc_id, "question": "What is the full name mentioned on the resume?"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert "Ashwin" in data["answer"]
    assert any(c["page"] == 1 for c in data["citations"])


# --- Broader / semantic questions (no exact keyword overlap) --------------

@requires_model
def test_answer_broad_semantic_question_ai_technologies(resume_doc_id):
    """The regression case: the question's wording ('AI technologies') does
    not literally appear in the source text ('RAG systems', 'large language
    models', 'embedding models', 'OCR pipelines'). This only passes with
    semantic retrieval, not literal keyword/TF-IDF matching."""
    resp = client.post(
        "/answer",
        json={"doc_id": resume_doc_id, "question": "What AI technologies are mentioned in the resume?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert len(data["citations"]) > 0
    # Should ground in the Skills page, which is where the AI/ML tooling is described.
    assert any(c["page"] == 3 for c in data["citations"])


@requires_model
def test_answer_broad_semantic_question_experience(resume_doc_id):
    resp = client.post(
        "/answer",
        json={"doc_id": resume_doc_id, "question": "What experiences are mentioned?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert len(data["citations"]) > 0
    assert any(c["page"] == 2 for c in data["citations"])


# --- Abstention -------------------------------------------------------------

@requires_model
def test_abstains_on_unanswerable_question(doc_id):
    resp = client.post(
        "/answer",
        json={"doc_id": doc_id, "question": "What is the company's stock ticker symbol on NASDAQ?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is True
    assert data["citations"] == []


@requires_model
def test_abstains_on_unrelated_question(resume_doc_id):
    """A question entirely unrelated to a resume's content."""
    resp = client.post(
        "/answer",
        json={"doc_id": resume_doc_id, "question": "What is the boiling point of mercury in Celsius?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is True
    assert data["citations"] == []


# --- Input validation --------------------------------------------------------

def test_unknown_doc_id_404():
    resp = client.post("/answer", json={"doc_id": "doesnotexist", "question": "Anything?"})
    assert resp.status_code == 404


def test_non_pdf_upload_rejected():
    resp = client.post("/ingest", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert resp.status_code == 400


@requires_model
def test_empty_question_rejected(doc_id):
    resp = client.post("/answer", json={"doc_id": doc_id, "question": "  "})
    assert resp.status_code == 400
