from __future__ import annotations
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingest import parse_pdf, chunk_pages
from app.store import registry
from app.answer import answer_question, AnswerResult

app = FastAPI(title="OneRx PDF RAG Service")


class IngestResponse(BaseModel):
    doc_id: str
    pages: int
    chunks: int


class AnswerRequest(BaseModel):
    doc_id: str
    question: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = uuid.uuid4().hex[:8]

    try:
        pages = parse_pdf(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}")

    chunks = chunk_pages(doc_id, pages)
    registry.add(doc_id, chunks)

    return IngestResponse(doc_id=doc_id, pages=len(pages), chunks=len(chunks))


@app.post("/answer", response_model=AnswerResult)
async def answer(req: AnswerRequest):
    index = registry.get(req.doc_id)
    if index is None:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {req.doc_id}")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    return answer_question(index, req.question)


@app.get("/health")
async def health():
    return {"status": "ok"}
