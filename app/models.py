"""
Pydantic data models for the RAG pipeline.
All request/response shapes are defined here for type safety and API documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class QueryIntent(str, Enum):
    """Classified intent of an incoming user query."""
    CONVERSATIONAL = "conversational"       # Greetings, small talk — no KB search needed
    KNOWLEDGE_SEARCH = "knowledge_search"   # Substantive question — triggers RAG pipeline
    PII_DETECTED = "pii_detected"           # Query contains personal info — refuse
    MEDICAL_DISCLAIMER = "medical_disclaimer"  # Medical/legal query — answer with disclaimer
    OUT_OF_SCOPE = "out_of_scope"           # Clearly unrelated to loaded documents


class ChunkMetadata(BaseModel):
    """Metadata attached to each document chunk during ingestion."""
    doc_id: str = Field(..., description="Unique identifier for the source document")
    filename: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="Page number this chunk came from (1-indexed)")
    chunk_index: int = Field(..., description="Position of this chunk within the document")
    total_chunks: int = Field(..., description="Total number of chunks in this document")
    char_start: int = Field(..., description="Character offset start in original document")
    char_end: int = Field(..., description="Character offset end in original document")


class DocumentChunk(BaseModel):
    """A single chunk of text with its embedding and metadata."""
    chunk_id: str = Field(..., description="Globally unique chunk identifier")
    text: str = Field(..., description="The actual text content of this chunk")
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = Field(
        default=None, description="Mistral embedding vector (1024 dimensions)"
    )


class IngestRequest(BaseModel):
    """Response model for the ingestion endpoint."""
    message: str
    documents_processed: int
    total_chunks: int
    doc_ids: list[str]


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval with its relevance scores."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    semantic_score: float = Field(..., description="Cosine similarity to query embedding")
    keyword_score: float = Field(..., description="BM25 keyword relevance score")
    hybrid_score: float = Field(..., description="Weighted combination of semantic + keyword")
    rerank_score: Optional[float] = Field(
        default=None, description="Score assigned by re-ranker (if applied)"
    )


class Citation(BaseModel):
    """A citation linking a claim in the answer to a source chunk."""
    chunk_id: str
    filename: str
    page_number: int
    relevant_excerpt: str = Field(..., description="The specific passage that supports the claim")
    confidence: float = Field(..., description="Similarity score for this citation (0-1)")


class QueryRequest(BaseModel):
    """Incoming query from the user."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score — chunks below this are discarded"
    )


class QueryResponse(BaseModel):
    """Full response from the query endpoint."""
    query: str
    intent: QueryIntent
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    confidence_score: float = Field(
        ..., description="Overall confidence in the answer (0-1), based on retrieval quality"
    )
    insufficient_evidence: bool = Field(
        ...,
        description="True if top chunks did not meet similarity threshold"
    )
    hallucination_flags: list[str] = Field(
        default_factory=list,
        description="Sentences in the answer that may not be supported by retrieved context"
    )
    disclaimer: Optional[str] = Field(
        default=None,
        description="Added for medical/legal queries"
    )
    processing_notes: list[str] = Field(
        default_factory=list,
        description="Internal pipeline notes for transparency"
    )


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str
    chunks_in_store: int
    documents_loaded: list[str]
    mistral_api_reachable: bool
