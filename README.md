# OneRx — PDF RAG Service

Ingest a PDF at runtime, answer questions grounded only in that document, with
citations, abstaining when the document doesn't support an answer.

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The requirements pin a CPU-only PyTorch wheel and a compatible Sentence
Transformers/Transformers stack; no CUDA, NVIDIA, torchvision, torchaudio, or
paid API is required.

No paid API key needed — the LLM is a local mock/stub behind an interface
(`app/llm.py`). Embeddings use a local Sentence Transformers model
(`sentence-transformers/all-MiniLM-L6-v2`). **First run downloads that model
(~90MB) from the Hugging Face Hub**, so an internet connection is required
once; after that it's cached locally (`~/.cache/huggingface`) and the service
runs fully offline with no per-query network calls.

If you're running in a network-restricted environment (no access to
`huggingface.co`), the model cannot load and `/ingest` and `/answer` will
fail — see "Known limitation" below.

## Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

## Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{ "doc_id": "abc12345", "pages": 12, "chunks": 84 }
```

## Ask a question

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "abc12345", "question": "What is the effective date of this policy?"}'
```

Response:
```json
{
  "answer": "...",
  "citations": [ { "page": 4, "chunk_id": "c-0031" } ],
  "abstained": false
}
```

If the document doesn't support an answer:
```json
{
  "answer": "The provided document does not contain information to answer this question.",
  "citations": [],
  "abstained": true
}
```

## Run tests

```bash
pytest tests/ -v
```

The service tests use the real embedding model when it is available. They are
not replaced with a fake embedder; in a network-restricted environment the
model-dependent tests explicitly skip. The model must be downloaded or cached
for the full end-to-end suite.

Two test files, split by whether they need the real embedding model:

- `tests/test_pipeline_logic.py` — unit tests for the extractive `MockLLM`
  and the retrieval/abstention gate in `app/answer.py`, using synthetic
  chunks and scores directly. No embedding model involved; always runs.
- `tests/test_service.py` — full end-to-end API tests against a real
  ingested PDF (built at test time by `tests/make_test_pdf.py`): specific
  fact questions, broader/semantic questions, citation correctness,
  abstention on unanswerable and unrelated questions, and input validation
  (bad doc_id, non-PDF upload, empty question). These require the real
  `all-MiniLM-L6-v2` model to be loadable (network access to
  `huggingface.co`, or an existing local cache). If the model can't load,
  each model-dependent test **skips with an explicit reason** rather than
  silently passing — check for `SKIPPED` vs `PASSED` in the output.

## Project layout

```
app/
  main.py    - FastAPI app, /ingest and /answer endpoints
  ingest.py  - PDF parsing (pypdf) and page-bounded sentence chunking
  store.py   - Per-document in-memory vector index (Sentence Transformers + FAISS)
  llm.py     - LLMClient interface + MockLLM extractive stub
  answer.py  - Retrieval, relevance gating, grounding, abstention
tests/
  test_service.py        - End-to-end API tests (require the real embedding model)
  test_pipeline_logic.py - Model-independent unit tests (MockLLM, abstention gate)
  make_test_pdf.py       - Test PDF fixture generators
DESIGN.md    - Design rationale
```

## Model availability

The first model load requires access to `huggingface.co` unless
`all-MiniLM-L6-v2` is already cached. Once cached, embeddings and inference
are local and queries make no network calls. The end-to-end tests use the real
model when it is available; in a network-restricted environment they skip
explicitly rather than using a fake embedder.

## Notes

- Storage is in-memory and process-lifetime (no persistence) — fine for the
  scope of this exercise; `doc_id`s are lost on restart.
- Swapping in a real LLM: implement `LLMClient` in `app/llm.py` and change
  `get_llm_client()`.
- Swapping the embedding model: change `MODEL_NAME` in `app/store.py`.
