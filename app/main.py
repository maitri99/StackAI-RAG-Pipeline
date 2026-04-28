"""
FastAPI RAG Pipeline — Main Application

Endpoints:
----------
POST /ingest          Upload one or more PDF files for ingestion
POST /query           Query the knowledge base with a natural language question
GET  /health          Health check and system status
DELETE /documents/{doc_id}  Remove a document from the knowledge base
GET  /documents       List all ingested documents

The application uses module-level singletons for the vector store and BM25 index,
meaning state is maintained in memory for the lifetime of the process.
"""

import os
import uuid
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.models import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    QueryIntent
)
from app.ingestion import ingest_pdf
from app.embeddings import (
    get_vector_store,
    get_embedder,
    embed_and_store_chunks
)
from app.retrieval import (
    get_bm25_index,
    rebuild_bm25_index,
    hybrid_search,
    rerank_chunks
)
from app.generation import (
    transform_query,
    generate_answer,
    generate_insufficient_evidence_response
)
from app.guardrails import (
    classify_intent,
    detect_pii,
    check_insufficient_evidence,
    detect_hallucinations,
    get_conversational_response,
    get_pii_refusal,
    get_out_of_scope_response
)

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY not found. "
        "Create a .env file with: MISTRAL_API_KEY=your_key_here"
    )

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "A production-grade Retrieval-Augmented Generation pipeline built from scratch. "
        "Supports PDF ingestion, hybrid semantic+keyword search, re-ranking, "
        "hallucination detection, and grounded answer generation using Mistral AI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS — allow the local UI to call the API ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the UI ──────────────────────────────────────────────────────────────
# Mount static files if the ui directory exists
import pathlib
ui_path = pathlib.Path("ui")
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the UI."""
    return FileResponse("ui/index.html")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and status",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Returns system status including:
    - Number of chunks currently in the vector store
    - List of ingested document filenames
    - Whether the Mistral API is reachable
    """
    store = get_vector_store()
    
    # Test Mistral API connectivity
    api_reachable = False
    try:
        embedder = get_embedder()
        api_reachable = embedder.test_connection()
    except Exception:
        api_reachable = False
    
    # Get unique filenames from ingested chunks
    loaded_docs = list(set(
        chunk.metadata.filename
        for chunk in store.chunks
    ))
    
    return HealthResponse(
        status="healthy" if api_reachable else "degraded",
        chunks_in_store=store.size,
        documents_loaded=loaded_docs,
        mistral_api_reachable=api_reachable
    )


# ── Ingestion Endpoint ────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestRequest,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest one or more PDF files",
    tags=["Ingestion"]
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="One or more PDF files to ingest")
) -> IngestRequest:
    """
    Upload and ingest PDF files into the knowledge base.
    
    Processing pipeline:
    1. Validate file types (PDF only)
    2. Extract text page-by-page using pdfminer
    3. Split text into overlapping chunks (recursive character splitting)
    4. Generate Mistral embeddings for each chunk
    5. Store chunks in in-memory vector store (NumPy)
    6. Rebuild BM25 keyword index
    
    Returns a summary of what was ingested.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided. Upload at least one PDF file."
        )
    
    store = get_vector_store()
    bm25_index = get_bm25_index()
    
    all_doc_ids = []
    total_chunks = 0
    processed_docs = 0
    errors = []
    
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            errors.append(f"{file.filename}: Only PDF files are supported.")
            continue
        
        try:
            # Read file bytes
            pdf_bytes = await file.read()
            
            if len(pdf_bytes) == 0:
                errors.append(f"{file.filename}: File is empty.")
                continue
            
            # Generate document ID
            doc_id = str(uuid.uuid4())[:8] + "_" + file.filename.replace('.pdf', '').replace(' ', '_')
            
            # Extract and chunk text
            chunks = ingest_pdf(
                pdf_bytes=pdf_bytes,
                filename=file.filename,
                doc_id=doc_id
            )
            
            if not chunks:
                errors.append(
                    f"{file.filename}: No text content could be extracted. "
                    "The PDF may be image-based (requires OCR)."
                )
                continue
            
            # Embed chunks and add to vector store
            embed_and_store_chunks(chunks)
            
            all_doc_ids.append(doc_id)
            total_chunks += len(chunks)
            processed_docs += 1
            
        except ValueError as e:
            errors.append(f"{file.filename}: {str(e)}")
        except Exception as e:
            errors.append(f"{file.filename}: Unexpected error — {str(e)}")
    
    # Rebuild BM25 index with all current chunks
    if processed_docs > 0:
        rebuild_bm25_index(store.chunks)
    
    if processed_docs == 0:
        error_detail = "; ".join(errors)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No documents were successfully ingested. Errors: {error_detail}"
        )
    
    message = f"Successfully ingested {processed_docs} document(s) into {total_chunks} chunks."
    if errors:
        message += f" Warnings: {'; '.join(errors)}"
    
    return IngestRequest(
        message=message,
        documents_processed=processed_docs,
        total_chunks=total_chunks,
        doc_ids=all_doc_ids
    )


# ── Query Endpoint ────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    tags=["Query"]
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Process a natural language query against the ingested knowledge base.
    
    Processing pipeline:
    1. Classify query intent (conversational, knowledge search, PII, etc.)
    2. Apply guardrails (PII refusal, out-of-scope detection)
    3. Transform query for better retrieval (HyDE)
    4. Embed the transformed query
    5. Hybrid search (semantic cosine similarity + BM25 keyword)
    6. Re-rank candidates using Mistral
    7. Check similarity threshold → return "insufficient evidence" if too low
    8. Generate grounded answer with citations
    9. Detect potential hallucinations in the answer
    10. Return structured response with all metadata
    """
    store = get_vector_store()
    processing_notes = []
    
    # ── Step 1: Intent classification ─────────────────────────────────────────
    intent = classify_intent(request.query)
    processing_notes.append(f"Intent classified as: {intent.value}")
    
    # ── Step 2: Guardrail responses ────────────────────────────────────────────
    if intent == QueryIntent.PII_DETECTED:
        pii_hits = detect_pii(request.query)
        return QueryResponse(
            query=request.query,
            intent=intent,
            answer=get_pii_refusal(pii_hits),
            citations=[],
            retrieved_chunks=[],
            confidence_score=0.0,
            insufficient_evidence=True,
            hallucination_flags=[],
            disclaimer=None,
            processing_notes=["Query refused: PII detected"]
        )
    
    if intent == QueryIntent.CONVERSATIONAL:
        return QueryResponse(
            query=request.query,
            intent=intent,
            answer=get_conversational_response(request.query),
            citations=[],
            retrieved_chunks=[],
            confidence_score=1.0,
            insufficient_evidence=False,
            hallucination_flags=[],
            disclaimer=None,
            processing_notes=["Conversational response — no KB search performed"]
        )
    
    if intent == QueryIntent.OUT_OF_SCOPE:
        return QueryResponse(
            query=request.query,
            intent=intent,
            answer=get_out_of_scope_response(),
            citations=[],
            retrieved_chunks=[],
            confidence_score=0.0,
            insufficient_evidence=True,
            hallucination_flags=[],
            disclaimer=None,
            processing_notes=["Query refused: out of scope"]
        )
    
    # ── Step 3: Check if knowledge base has content ────────────────────────────
    if store.size == 0:
        return QueryResponse(
            query=request.query,
            intent=intent,
            answer=(
                "No documents have been ingested yet. "
                "Please upload PDF documents using the /ingest endpoint first."
            ),
            citations=[],
            retrieved_chunks=[],
            confidence_score=0.0,
            insufficient_evidence=True,
            hallucination_flags=[],
            disclaimer=None,
            processing_notes=["No documents in knowledge base"]
        )
    
    # ── Step 4: Query transformation ──────────────────────────────────────────
    transformed_query = transform_query(request.query, MISTRAL_API_KEY)
    if transformed_query != request.query:
        processing_notes.append("Query transformed using HyDE for better retrieval")
    
    # ── Step 5: Embed transformed query ───────────────────────────────────────
    try:
        embedder = get_embedder()
        query_embedding = embedder.embed_query(transformed_query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}"
        )
    
    # ── Step 6: Hybrid search ──────────────────────────────────────────────────
    bm25_index = get_bm25_index()
    candidates_k = min(request.top_k * 2, store.size)
    
    retrieved_chunks = hybrid_search(
        query_embedding=query_embedding,
        query_text=request.query,
        vector_store=store,
        bm25_index=bm25_index,
        top_k=candidates_k,
        similarity_threshold=0.0  # Apply threshold after re-ranking
    )
    processing_notes.append(
        f"Hybrid search retrieved {len(retrieved_chunks)} candidate chunks"
    )
    
    # ── Step 7: Re-rank candidates ─────────────────────────────────────────────
    if len(retrieved_chunks) > 1:
        retrieved_chunks = rerank_chunks(
            query=request.query,
            chunks=retrieved_chunks,
            api_key=MISTRAL_API_KEY,
            top_k=request.top_k
        )
        processing_notes.append(f"Re-ranked to top {len(retrieved_chunks)} chunks")
    
    # ── Step 8: Threshold check ────────────────────────────────────────────────
    insufficient = check_insufficient_evidence(
        retrieved_chunks,
        threshold=request.similarity_threshold
    )
    
    if insufficient:
        best_score = max(
            (c.rerank_score or c.hybrid_score for c in retrieved_chunks),
            default=0.0
        )
        processing_notes.append(
            f"Insufficient evidence: best score {best_score:.3f} < "
            f"threshold {request.similarity_threshold:.3f}"
        )
        return QueryResponse(
            query=request.query,
            intent=intent,
            answer=generate_insufficient_evidence_response(
                request.query, best_score, request.similarity_threshold
            ),
            citations=[],
            retrieved_chunks=retrieved_chunks,
            confidence_score=best_score,
            insufficient_evidence=True,
            hallucination_flags=[],
            disclaimer=None,
            processing_notes=processing_notes
        )
    
    # ── Step 9: Generate answer ────────────────────────────────────────────────
    add_disclaimer = (intent == QueryIntent.MEDICAL_DISCLAIMER)
    
    answer, citations = generate_answer(
        query=request.query,
        chunks=retrieved_chunks,
        api_key=MISTRAL_API_KEY,
        intent=intent,
        add_disclaimer=add_disclaimer
    )
    
    # ── Step 10: Hallucination detection ──────────────────────────────────────
    context_texts = [chunk.text for chunk in retrieved_chunks]
    hallucination_flags = detect_hallucinations(answer, context_texts)
    
    if hallucination_flags:
        processing_notes.append(
            f"Hallucination check: {len(hallucination_flags)} potentially unsupported claim(s) flagged"
        )
    
    # ── Compute overall confidence score ──────────────────────────────────────
    if retrieved_chunks:
        top_score = max(c.rerank_score or c.hybrid_score for c in retrieved_chunks)
        hallucination_penalty = len(hallucination_flags) * 0.05
        confidence_score = max(0.0, min(1.0, top_score - hallucination_penalty))
    else:
        confidence_score = 0.0
    
    disclaimer = (
        "⚠️ Medical/Legal Disclaimer: This response is for informational purposes only "
        "and does not constitute professional medical or legal advice."
        if add_disclaimer else None
    )
    
    return QueryResponse(
        query=request.query,
        intent=intent,
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        confidence_score=round(confidence_score, 4),
        insufficient_evidence=False,
        hallucination_flags=hallucination_flags,
        disclaimer=disclaimer,
        processing_notes=processing_notes
    )


# ── Document Management Endpoints ─────────────────────────────────────────────

@app.get(
    "/documents",
    summary="List all ingested documents",
    tags=["Documents"]
)
async def list_documents():
    """Return a list of all currently ingested documents."""
    store = get_vector_store()
    
    doc_summary = {}
    for chunk in store.chunks:
        doc_id = chunk.metadata.doc_id
        if doc_id not in doc_summary:
            doc_summary[doc_id] = {
                "doc_id": doc_id,
                "filename": chunk.metadata.filename,
                "chunk_count": 0,
                "page_count": 0
            }
        doc_summary[doc_id]["chunk_count"] += 1
        doc_summary[doc_id]["page_count"] = max(
            doc_summary[doc_id]["page_count"],
            chunk.metadata.page_number
        )
    
    return {
        "total_documents": len(doc_summary),
        "total_chunks": store.size,
        "documents": list(doc_summary.values())
    }


@app.delete(
    "/documents/{doc_id}",
    summary="Remove a document from the knowledge base",
    tags=["Documents"]
)
async def delete_document(doc_id: str):
    """Remove all chunks for the specified document from the knowledge base."""
    store = get_vector_store()
    
    removed = store.remove_document(doc_id)
    
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in the knowledge base."
        )
    
    # Rebuild BM25 index after deletion
    rebuild_bm25_index(store.chunks)
    
    return {
        "message": f"Successfully removed {removed} chunks for document '{doc_id}'.",
        "chunks_removed": removed,
        "chunks_remaining": store.size
    }
