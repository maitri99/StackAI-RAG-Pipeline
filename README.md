# RAG Pipeline - Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch using **FastAPI** and **Mistral AI**. No external RAG libraries (LangChain, LlamaIndex, Haystack) or third-party vector databases are used - all retrieval logic is implemented directly.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Component Design Decisions](#component-design-decisions)
- [API Endpoints](#api-endpoints)
- [How to Run](#how-to-run)
- [Running Tests](#running-tests)
- [Libraries Used](#libraries-used)
- [Bonus Features](#bonus-features)

---

## System Overview

RAG augments a language model by retrieving relevant passages from a document corpus before generating an answer. Rather than relying on the LLM's training data (which may be outdated or hallucinated), the model is constrained to answer only from retrieved evidence.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
│                                                                 │
│  Intent Detection → Query Transform → Embed → Hybrid Search     │
│       │                                            │            │
│  [Guardrails]                              Re-rank Candidates   │
│  PII / OOScope                                     │            │
│  Medical disclaimer                        Threshold Check      │
│                                                    │            │
│                                          Generate Answer        │
│                                          (grounded generation)  │
│                                                    │            │
│                                          Hallucination Scan     │
│                                                    │            │
│                                            Response + Citations │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Ingestion Pipeline (`/ingest`)

```
PDF File(s)
    │
    ▼
pdfminer.six          ← text extraction, page-by-page
    │
    ▼
Recursive Chunker     ← paragraph → sentence → word boundary splitting
    │                    chunk_size=1800 chars, overlap=200 chars
    ▼
Mistral Embeddings    ← mistral-embed model, 1024 dimensions, batched
    │
    ▼
NumPy Vector Store    ← L2-normalized (N×1024) float32 matrix
    +
BM25 Inverted Index   ← keyword index rebuilt after each ingestion
```

### Query Pipeline (`/query`)

```
User Query
    │
    ├─► Intent Classifier ──► CONVERSATIONAL → direct response
    │        │                PII_DETECTED   → refuse + explain
    │        │                OUT_OF_SCOPE   → gentle refusal
    │        │                MEDICAL        → answer + disclaimer
    │        │                KNOWLEDGE      → continue pipeline
    │
    ▼
Query Transformer (HyDE)
    │    Generates hypothetical answer passage for better semantic matching
    │
    ▼
Mistral Embeddings    ← embed transformed query
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
Semantic Search                      BM25 Keyword Search
(cosine similarity                   (handles exact terms,
 on NumPy matrix)                    codes, proper nouns)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
          Hybrid Score Fusion
          (70% semantic + 30% keyword)
                   │
                   ▼
          Re-ranker (Mistral)
          Cross-encoder scoring, top-K final
                   │
                   ▼
          Similarity Threshold Check
          → "Insufficient Evidence" if below threshold
                   │
                   ▼
          Grounded Generation (Mistral)
          System prompt enforces citation-only answers
                   │
                   ▼
          Hallucination Detection
          Scans answer for unsupported factual claims
                   │
                   ▼
          Structured Response
          (answer + citations + confidence + flags)
```

---

## Component Design Decisions

### 1. Text Extraction: pdfminer.six

`pdfminer.six` was chosen over PyPDF2 for its superior handling of complex layouts (multi-column, tables), font encoding edge cases, and better whitespace preservation. Text is extracted page-by-page to preserve page number metadata for citations.

### 2. Chunking Strategy: Recursive Character Splitting with Overlap

**Chunk size: 1800 characters (~450 tokens)**

- Too large: retrieved chunks contain irrelevant text, diluting the signal sent to the LLM
- Too small: chunks lose surrounding context needed to answer questions spanning multiple sentences

**Overlap: 200 characters (≈11%)**

Ensures sentences split across chunk boundaries are still retrievable from both sides. Critical for questions whose answers span paragraph breaks.

**Splitting hierarchy: paragraph → sentence → word → character**

The splitter tries natural linguistic boundaries before hard-cutting mid-sentence. This preserves coherence within each chunk and improves embedding quality.

### 3. Embeddings: Mistral Embed (1024 dimensions)

Mistral's dedicated embedding model is optimized for retrieval tasks. Embeddings are generated in batches of 32 to balance throughput and API rate limits.

### 4. Vector Store: Pure NumPy (No External Database)

All embeddings are stored as a pre-normalized `(N × 1024)` float32 NumPy matrix. Cosine similarity at query time is a single matrix-vector multiplication:

```
similarity = embeddings_matrix @ query_vector
```

Pre-normalizing all vectors at ingestion time means the dot product equals cosine similarity, making query-time search very fast. This intentionally replaces external vector databases (Pinecone, Qdrant, Chroma) to demonstrate the underlying retrieval math directly.

**Production path:** For >100k chunks, swap the NumPy matrix for FAISS or Qdrant without changing any other component - the VectorStore interface stays identical.

### 5. Hybrid Search: Semantic + BM25

Pure semantic search fails on exact terms that have no semantic neighborhood: medical codes (ICD-10, HCPCS), drug names, abbreviations, and proper nouns. BM25 excels at exactly these cases.

**BM25 implementation from scratch:**

```
score(q, d) = Σ IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1-b + b×|d|/avgdl)]
```

Where `k1=1.5` (term frequency saturation) and `b=0.75` (document length normalization).

**Score fusion:**

```
hybrid_score = 0.70 × semantic_normalized + 0.30 × bm25_normalized
```

Both scores are min-max normalized to [0,1] before fusion. Semantic is weighted higher because it captures meaning, while BM25 captures exact matches.

### 6. Query Transformation (HyDE)

Raw user queries are often short and stylistically different from document text. HyDE (Hypothetical Document Embeddings) generates a short hypothetical answer passage using the LLM, then embeds that passage for retrieval. This bridges the semantic gap between query style and document style.

### 7. Re-ranking: Cross-Encoder Pattern

After retrieving `top_k × 2` candidates via hybrid search, a re-ranking step asks Mistral to score each chunk's relevance to the specific query on a 0-10 scale. This cross-encoder pattern allows query-document interaction (unlike bi-encoders which encode independently) and significantly improves precision for the final top-K results.

### 8. Grounded Generation

The generation system prompt strictly instructs the LLM:
- Answer ONLY from the provided context
- Cite every factual claim with `[SOURCE_N]` markers
- If the context is insufficient, say so explicitly - never fabricate

This prevents the LLM from using its training data as a knowledge source.

### 9. Hallucination Detection

After generation, the answer is scanned for sentences containing specific factual claims (numbers, percentages, dollar amounts, proper nouns, quoted phrases). Each claim is checked against the retrieved context - claims whose key terms are absent from the context are flagged as potentially unsupported.

This is a heuristic approach. Production systems would use a dedicated NLI (Natural Language Inference) model for entailment checking.

---

## API Endpoints

### `POST /ingest`
Upload one or more PDF files for ingestion into the knowledge base.

**Request:** `multipart/form-data` with one or more PDF files

**Response:**
```json
{
  "message": "Successfully ingested 1 document(s) into 42 chunks.",
  "documents_processed": 1,
  "total_chunks": 42,
  "doc_ids": ["abc123_my_document"]
}
```

---

### `POST /query`
Query the knowledge base with a natural language question.

**Request:**
```json
{
  "query": "What are the prior authorization requirements for adalimumab?",
  "top_k": 5,
  "similarity_threshold": 0.45
}
```

**Response:**
```json
{
  "query": "What are the prior authorization requirements for adalimumab?",
  "intent": "knowledge_search",
  "answer": "The prior authorization requirements include... [SOURCE_1]",
  "citations": [
    {
      "chunk_id": "...",
      "filename": "policy.pdf",
      "page_number": 3,
      "relevant_excerpt": "...",
      "confidence": 0.87
    }
  ],
  "retrieved_chunks": [...],
  "confidence_score": 0.87,
  "insufficient_evidence": false,
  "hallucination_flags": [],
  "disclaimer": null,
  "processing_notes": [
    "Intent classified as: knowledge_search",
    "Query transformed using HyDE for better retrieval",
    "Hybrid search retrieved 10 candidate chunks",
    "Re-ranked to top 5 chunks"
  ]
}
```

---

### `GET /health`
Returns system status.

```json
{
  "status": "healthy",
  "chunks_in_store": 42,
  "documents_loaded": ["clinical_notes.pdf"],
  "mistral_api_reachable": true
}
```

---

### `GET /documents`
List all ingested documents with chunk counts.

### `DELETE /documents/{doc_id}`
Remove a document and all its chunks from the knowledge base.

---

## How to Run

### Prerequisites
- Python 3.11+
- Anaconda or pip
- Mistral AI API key (free at https://console.mistral.ai/)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

# 2. Create and activate virtual environment
conda create -n rag-pipeline python=3.11 -y
conda activate rag-pipeline

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your Mistral API key:
# MISTRAL_API_KEY=your_key_here

# 5. Start the server
uvicorn app.main:app --reload

# 6. Open the UI
# Navigate to http://localhost:8000 in your browser
```

### Using the API directly

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "files=@your_document.pdf"

# Query the knowledge base
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "top_k": 5}'

# Check health
curl http://localhost:8000/health
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Text chunking logic (size, overlap, minimum chunk enforcement)
- BM25 index construction and scoring
- Vector store operations (add, search, remove, clear)
- PII detection (SSN, email, phone, MRN)
- Intent classification (conversational, knowledge, PII, medical, OOS)
- Hallucination detection heuristic

---

## Libraries Used

| Library | Version | Purpose | Link |
|---------|---------|---------|------|
| FastAPI | 0.115.0 | Web framework and API layer | https://fastapi.tiangolo.com |
| Uvicorn | 0.30.6 | ASGI server | https://www.uvicorn.org |
| pdfminer.six | 20231228 | PDF text extraction | https://pdfminersix.readthedocs.io |
| NumPy | 1.26.4 | Vector storage and cosine similarity | https://numpy.org |
| Requests | 2.32.3 | Mistral API calls | https://requests.readthedocs.io |
| python-dotenv | 1.0.1 | Environment variable management | https://pypi.org/project/python-dotenv |
| python-multipart | 0.0.9 | File upload support in FastAPI | https://pypi.org/project/python-multipart |
| Pydantic | 2.8.2 | Data validation and API models | https://docs.pydantic.dev |
| pytest | 8.3.2 | Testing framework | https://pytest.org |
| httpx | 0.27.2 | Async HTTP client for tests | https://www.python-httpx.org |
| Mistral AI API | - | Embeddings (mistral-embed) + Generation (mistral-small-latest) | https://docs.mistral.ai |

**Intentionally not used:**
- ~~LangChain~~ - all retrieval logic implemented from scratch
- ~~LlamaIndex~~ - no external RAG framework
- ~~Haystack~~ - no external RAG framework
- ~~Pinecone / Qdrant / Chroma~~ - NumPy vector store used instead
- ~~FAISS~~ - pure NumPy cosine similarity used instead

---

## Bonus Features

### Insufficient Evidence Refusal
If the maximum similarity score across all retrieved chunks falls below the configurable threshold (default: 0.45), the system returns an informative refusal rather than hallucinating an answer. The response explains the gap and suggests remediation.

### Answer Shaping by Intent
The generation prompt adapts its output format based on detected query intent:
- List queries ("what are...", "list all...") → bulleted list format
- Comparison queries ("compare", "difference between") → markdown table
- Process queries ("how to", "steps for") → numbered steps
- Default → prose paragraphs

### Hallucination Filtering
Post-generation scan checks specific factual claims (numbers, percentages, proper nouns, quoted phrases) against retrieved context. Unsupported claims are flagged in the response as `hallucination_flags`.

### Query Refusal Policies
- **PII detection:** Regex patterns detect SSN, credit cards, phone numbers, emails, MRNs before the query reaches any API
- **Medical/legal disclaimer:** Queries seeking personal medical or legal advice receive a mandatory disclaimer prepended to the answer
- **Out of scope:** Queries unrelated to document content receive a gentle refusal without consuming API credits
