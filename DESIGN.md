# DESIGN

## Chunking strategy

PDF text is extracted **per page** using `pypdf`. Each page is whitespace-normalized and split into sentence-boundary-aware chunks of approximately **250 characters maximum**. Chunks never cross page boundaries, so every chunk has an unambiguous source page. Sentence-aware splitting keeps individual facts and statements intact while producing smaller retrieval units than page-level chunks.

Each chunk retains `doc_id`, `page`, `chunk_id`, and `text` metadata for retrieval and citation.

## Embeddings / vector store

The service uses the Sentence Transformers model **`all-MiniLM-L6-v2`** to create 384-dimensional semantic embeddings locally. Embeddings are L2-normalized and stored in a per-document **FAISS `IndexFlatIP`** index. Because the vectors are normalized, inner-product search is equivalent to cosine-similarity search.

The index is held in memory for the lifetime of the process, which is sufficient for the scope of this exercise. The model is downloaded once from Hugging Face when first required and then cached locally. No paid embedding API is used.

## Grounding enforcement

Grounding is enforced in two stages:

1. **Retrieval filtering:** the question is embedded and the top candidate chunks are retrieved using FAISS similarity search. A conservative relevance floor of **0.28** prevents clearly unrelated questions from reaching the answering stage. Relevant nearby results can also be retained within a bounded score band around a strong result, and bounded same-page corroboration is allowed so that useful supporting context is not discarded solely because it has a slightly lower score.

2. **Extractive answering:** the `LLMClient` interface is implemented by `MockLLM` for this assignment. It receives only the chunks that passed the retrieval gate and selects sentences **verbatim from those chunks**. It does not generate facts or use outside knowledge. Lexical overlap is used only to rank candidate sentences; semantic relevance is determined by the embedding retrieval stage.

Citations are generated from the chunks whose sentences actually appear in the final answer rather than from every retrieved chunk. This keeps each citation tied to supporting source material.

## Abstention

If retrieval does not produce sufficiently relevant evidence, the service abstains instead of attempting to answer. The relevance floor and bounded corroboration rules are deliberately conservative. An unsupported or unrelated question therefore returns:

```json
{
  "answer": "The provided document does not contain information to answer this question.",
  "citations": [],
  "abstained": true
}
```

If retrieval succeeds but the extractive answering step cannot select any source text, the service also abstains. There is no fallback to external knowledge or free-form generation, so unsupported answers cannot be fabricated by the mock LLM.

## Why this design

The original implementation used larger page-level character chunks and a strict top-score gate. That made precise facts less focused in embeddings and caused broader semantic questions to lose nearby relevant evidence. The final design uses smaller sentence-aware chunks plus conservative corroboration to improve both precise and broad retrieval while preserving abstention for unsupported questions.

The implementation remains intentionally small and aligned with the assignment scope: FastAPI, runtime PDF parsing, Sentence Transformers, FAISS, an LLM interface with a local mock implementation, grounded citations, and explicit abstention. No external paid LLM, API key, CUDA/NVIDIA dependency, TF-IDF retrieval, or document-specific hardcoding is required.
