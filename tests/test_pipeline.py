"""
Tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestChunking:
    """Tests for the text chunking logic."""
    
    def test_short_text_returns_single_chunk(self):
        from app.ingestion import _recursive_split
        text = "This is a short paragraph that fits in one chunk."
        chunks = _recursive_split(text, chunk_size=1800, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        from app.ingestion import _recursive_split
        # Create text definitely larger than one chunk
        text = ("This is a test sentence. " * 100)
        chunks = _recursive_split(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_minimum_size(self):
        from app.ingestion import _recursive_split, MIN_CHUNK_SIZE
        text = "Short. " * 50
        chunks = _recursive_split(text, chunk_size=500, overlap=50)
        for chunk in chunks:
            assert len(chunk) >= MIN_CHUNK_SIZE

    def test_overlap_creates_continuity(self):
        from app.ingestion import _recursive_split
        text = "word " * 400  # 2000 chars
        chunks = _recursive_split(text, chunk_size=500, overlap=100)
        # Check that consecutive chunks share some content (overlap)
        if len(chunks) >= 2:
            # Last 100 chars of chunk 0 should appear in chunk 1
            end_of_first = chunks[0][-50:]
            assert end_of_first in chunks[1] or len(chunks[0]) < 500

    def test_clean_text_removes_control_chars(self):
        from app.ingestion import _clean_text
        dirty = "Hello\x00World\x01Test\n\nParagraph"
        clean = _clean_text(dirty)
        assert '\x00' not in clean
        assert '\x01' not in clean
        assert "Hello" in clean
        assert "World" in clean


# ── Guardrails tests ──────────────────────────────────────────────────────────

class TestGuardrails:
    """Tests for PII detection and intent classification."""
    
    def test_pii_ssn_detected(self):
        from app.guardrails import detect_pii
        hits = detect_pii("My SSN is 123-45-6789")
        assert len(hits) > 0
        assert any("Social Security" in pii_type for _, pii_type in hits)

    def test_pii_email_detected(self):
        from app.guardrails import detect_pii
        hits = detect_pii("Contact me at john.doe@example.com")
        assert len(hits) > 0

    def test_pii_phone_detected(self):
        from app.guardrails import detect_pii
        hits = detect_pii("Call me at (555) 123-4567")
        assert len(hits) > 0

    def test_clean_query_no_pii(self):
        from app.guardrails import detect_pii
        hits = detect_pii("What are the prior authorization criteria for adalimumab?")
        assert len(hits) == 0

    def test_conversational_intent(self):
        from app.guardrails import classify_intent
        from app.models import QueryIntent
        assert classify_intent("Hello!") == QueryIntent.CONVERSATIONAL
        assert classify_intent("hi") == QueryIntent.CONVERSATIONAL
        assert classify_intent("thanks") == QueryIntent.CONVERSATIONAL

    def test_knowledge_search_intent(self):
        from app.guardrails import classify_intent
        from app.models import QueryIntent
        intent = classify_intent("What are the prior authorization requirements for Humira?")
        assert intent == QueryIntent.KNOWLEDGE_SEARCH

    def test_pii_intent_overrides_all(self):
        from app.guardrails import classify_intent
        from app.models import QueryIntent
        # Even a knowledge query with SSN should be refused
        intent = classify_intent("Tell me about 123-45-6789 and their treatment")
        assert intent == QueryIntent.PII_DETECTED

    def test_medical_disclaimer_intent(self):
        from app.guardrails import classify_intent
        from app.models import QueryIntent
        intent = classify_intent("Should I take this medication for my symptoms?")
        assert intent == QueryIntent.MEDICAL_DISCLAIMER


# ── BM25 tests ────────────────────────────────────────────────────────────────

class TestBM25:
    """Tests for the BM25 keyword search implementation."""
    
    def _make_chunks(self, texts):
        """Helper to create mock DocumentChunk objects."""
        from app.models import DocumentChunk, ChunkMetadata
        import uuid
        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text,
                metadata=ChunkMetadata(
                    doc_id=f"doc_{i}",
                    filename=f"test_{i}.pdf",
                    page_number=1,
                    chunk_index=i,
                    total_chunks=len(texts),
                    char_start=0,
                    char_end=len(text)
                )
            )
            chunks.append(chunk)
        return chunks

    def test_exact_match_scores_higher(self):
        from app.retrieval import BM25Index
        chunks = self._make_chunks([
            "prior authorization requirements for adalimumab",
            "weather forecast for tomorrow sunny skies",
            "treatment guidelines for Crohn's disease"
        ])
        index = BM25Index()
        index.build(chunks)
        scores = index.get_scores("prior authorization adalimumab")
        # Chunk 0 should score highest
        assert scores.get(0, 0) > scores.get(1, 0)

    def test_empty_query_returns_no_scores(self):
        from app.retrieval import BM25Index
        chunks = self._make_chunks(["Some content here"])
        index = BM25Index()
        index.build(chunks)
        scores = index.get_scores("")
        assert len(scores) == 0

    def test_build_with_empty_chunks(self):
        from app.retrieval import BM25Index
        index = BM25Index()
        index.build([])
        scores = index.get_scores("test query")
        assert scores == {}


# ── Vector Store tests ────────────────────────────────────────────────────────

class TestVectorStore:
    """Tests for the in-memory NumPy vector store."""
    
    def _make_embedded_chunk(self, text: str, embedding: list):
        from app.models import DocumentChunk, ChunkMetadata
        import uuid
        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            metadata=ChunkMetadata(
                doc_id="test_doc",
                filename="test.pdf",
                page_number=1,
                chunk_index=0,
                total_chunks=1,
                char_start=0,
                char_end=len(text)
            ),
            embedding=embedding
        )

    def test_add_chunks_increases_size(self):
        from app.embeddings import VectorStore
        store = VectorStore()
        dim = 8
        chunk = self._make_embedded_chunk("test text", [0.1] * dim)
        store.add_chunks([chunk])
        assert store.size == 1

    def test_cosine_similarity_returns_correct_count(self):
        from app.embeddings import VectorStore
        store = VectorStore()
        dim = 8
        chunks = [
            self._make_embedded_chunk(f"chunk {i}", [float(i)] * dim)
            for i in range(1, 6)
        ]
        store.add_chunks(chunks)
        results = store.cosine_similarity_search([1.0] * dim, top_k=3)
        assert len(results) == 3

    def test_identical_vectors_score_one(self):
        from app.embeddings import VectorStore
        store = VectorStore()
        vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        chunk = self._make_embedded_chunk("test", vec)
        store.add_chunks([chunk])
        results = store.cosine_similarity_search(vec, top_k=1)
        assert len(results) == 1
        assert abs(results[0][1] - 1.0) < 1e-5  # Should be ~1.0

    def test_clear_empties_store(self):
        from app.embeddings import VectorStore
        store = VectorStore()
        chunk = self._make_embedded_chunk("test", [0.1] * 8)
        store.add_chunks([chunk])
        store.clear()
        assert store.size == 0

    def test_remove_document(self):
        from app.embeddings import VectorStore
        from app.models import DocumentChunk, ChunkMetadata
        import uuid
        
        store = VectorStore()
        
        # Add chunks from two different documents
        for doc_id in ["doc_a", "doc_b"]:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=f"Content from {doc_id}",
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    filename=f"{doc_id}.pdf",
                    page_number=1,
                    chunk_index=0,
                    total_chunks=1,
                    char_start=0,
                    char_end=20
                ),
                embedding=[0.1] * 8
            )
            store.add_chunks([chunk])
        
        assert store.size == 2
        store.remove_document("doc_a")
        assert store.size == 1
        assert "doc_b" in store.document_ids
        assert "doc_a" not in store.document_ids


# ── Hallucination detection tests ─────────────────────────────────────────────

class TestHallucinationDetection:
    """Tests for the hallucination detection heuristic."""
    
    def test_supported_claim_not_flagged(self):
        from app.guardrails import detect_hallucinations
        answer = "The approval rate is 87% according to the study."
        context = ["The approval rate is 87% according to the study conducted in 2024."]
        flags = detect_hallucinations(answer, context)
        assert len(flags) == 0

    def test_unsupported_number_flagged(self):
        from app.guardrails import detect_hallucinations
        answer = "The medication costs $50,000 per year."
        context = ["The medication has been approved for Crohn's disease treatment."]
        flags = detect_hallucinations(answer, context)
        assert len(flags) > 0

    def test_empty_answer_no_flags(self):
        from app.guardrails import detect_hallucinations
        flags = detect_hallucinations("", ["some context"])
        assert flags == []

    def test_empty_context_no_crash(self):
        from app.guardrails import detect_hallucinations
        flags = detect_hallucinations("Some answer with 42% statistic.", [])
        # May or may not flag, but should not crash
        assert isinstance(flags, list)
