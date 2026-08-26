# DESIGN

## Chunking strategy
PDF text is extracted per-page (`pypdf`), whitespace-normalized, then split
into sentence-boundary-aware groups of up to ~250 characters. Chunks never cross a
page boundary — each chunk therefore has exactly one, unambiguous page
number, which keeps citations honest and simple. Sentence boundaries avoid
cutting facts through the middle of a sentence. Chunk size was chosen
empirically as a reasonable balance for a small extractive stub: large
enough to hold a complete sentence/fact, small enough to keep retrieval
precise.

## Embedding / vector-store choice
Embeddings: **Sentence Transformers**, model `all-MiniLM-L6-v2`
(384-dimensional), run locally via the `sentence-transformers` package —
no paid API key, no per-query network call. The model downloads once from
the Hugging Face Hub on first use (~90MB) and is cached locally afterward,
so the service runs fully offline after that first download.

Vector store: **FAISS** `IndexFlatIP` — one index per document, built at
ingest time, held in memory for the process lifetime. Chunk embeddings are
L2-normalized before insertion, so inner product on `IndexFlatIP` is
equivalent to cosine similarity.

An earlier version of this service used TF-IDF instead of a neural
embedding model, specifically to avoid any model download. That was
dropped: TF-IDF only matches shared vocabulary, so a question like *"What
AI technologies are mentioned?"* fails to retrieve a chunk that says *"built
RAG pipelines using LangChain and vector search"* — there's no literal word
overlap, even though it's exactly the right chunk. A sentence embedding
model captures that semantic relationship instead of requiring the
question and the answer to share words.

Every retrieved result still carries `doc_id`, `page`, `chunk_id`, and
`text` (see `app/ingest.py`'s `Chunk` dataclass and `app/store.py`'s
`DocumentIndex.search`), so citation metadata survives the embedding change
unchanged.

## Grounding enforcement
Two layers, both required for a non-abstained answer:

1. **Retrieval gate** — top-k chunks (k=5) are retrieved by cosine
   similarity against the MiniLM embedding of the question; only chunks
   the best score must be ≥ 0.28; nearby results within 0.20 of that best
   score are also retained as candidate context. Below the absolute floor,
   the question and the document are considered unrelated. This threshold
   is calibrated for MiniLM's cosine-similarity range (genuinely related
   short texts commonly score ~0.3–0.6+; unrelated text is usually well
   under 0.28) — it is a different constant than the TF-IDF version used,
   because TF-IDF and neural cosine similarity have very different score
   distributions. It is deliberately not set near zero: a low-enough
   threshold would let the system "answer" questions the document doesn't
   actually support, which the assignment explicitly rules out.
2. **Extractive answering** — the "LLM" (`MockLLM`, behind an `LLMClient`
   interface so a real model can be swapped in later) does not generate
   free text. It receives only the chunks that cleared the retrieval gate,
   and returns *verbatim sentences* from them, ranked by lexical overlap
   with the question (used only to pick which sentences to surface, not as
   a second relevance gate — relevance was already decided by retrieval).
   Because the answer is copy-pasted from retrieved text, no fact can be
   invented — grounding is enforced by construction, not by prompting a
   generative model to "only use the context" and hoping.

Citations are derived from exactly which chunks contributed the sentences
used in the answer (not all retrieved chunks), so every citation actually
supports a piece of the answer.

## Abstention logic
"Not in the document" is decided at the retrieval gate described above: if
the top retrieval score is below the absolute floor and there is no bounded
same-page corroboration, or nothing is retrieved at all, the pipeline returns
`abstained: true`, an empty
citations list, and a fixed safe message — and never reaches the answering
step, so there is no path by which the mock LLM can output invented
content. If retrieval finds relevant chunks but the extraction step somehow
produces no text (e.g. empty chunk list edge case), that also abstains
rather than falling back to any kind of default or generated answer.

## Change log vs. the original TF-IDF version
- `app/store.py`: TF-IDF vectorizer → `SentenceTransformer('all-MiniLM-L6-v2')`,
  `IndexFlatIP` retained.
- `app/answer.py`: similarity threshold is calibrated to 0.28 for normalized
  MiniLM cosine scores, with bounded same-page support for lower-score
  clusters and a 0.20 score band around a strong best match.
- `app/llm.py`: `MockLLM` previously required literal keyword overlap
  between the question and a candidate sentence as a hard pass/fail gate.
  That double-gated relevance on top of retrieval and caused correct,
  semantically-retrieved chunks to be rejected at the extraction step for
  broad/paraphrased questions. Lexical overlap is now used only to *rank*
  candidate sentences; if none have any overlap, the top-ranked (i.e. most
  semantically relevant per retrieval) chunk's own sentences are used as a
  fallback — still 100% drawn from retrieved text, never generated.
