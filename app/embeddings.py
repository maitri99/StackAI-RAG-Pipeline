"""
Embeddings and In-Memory Vector Store

Design Decisions:
-----------------
1. EMBEDDING MODEL: mistral-embed (1024 dimensions)
   - Mistral's dedicated embedding model, optimized for retrieval tasks
   - 1024 dimensions balances quality vs. memory footprint
   - Batch embedding (up to 512 texts per API call) reduces latency and cost

2. VECTOR STORE: Pure NumPy — no external vector database
   - Embeddings stored as a (N x 1024) float32 numpy matrix
   - Cosine similarity computed via matrix multiplication: O(N*D) per query
   - This is intentionally simple — demonstrates understanding of the underlying
     math rather than hiding it behind a library
   - For production scale (>100k chunks), you'd swap this for FAISS or Qdrant
     without changing any other component (the interface stays the same)

3. COSINE SIMILARITY: dot product of L2-normalized vectors
   - Measures angle between vectors, not magnitude — ideal for text similarity
   - Pre-normalizing all vectors at ingestion time means query-time similarity
     is a single matrix multiplication (very fast)

4. PERSISTENCE: In-memory only during this demo. In production, serialize
   the numpy matrix + metadata to disk (numpy .npy format) or a proper store.
"""

import os
import time
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple

from app.models import DocumentChunk, RetrievedChunk


# ── Constants ─────────────────────────────────────────────────────────────────
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 32          # Texts per embedding API call (conservative to avoid timeouts)
MAX_RETRIES = 3          # Retry failed API calls with exponential backoff


class MistralEmbedder:
    """
    Wraps the Mistral embedding API with batching, retry logic, and rate limit handling.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the Mistral embedding API.
        Handles batching and retries automatically.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start:batch_start + BATCH_SIZE]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch with exponential backoff retry."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{MISTRAL_API_BASE}/embeddings",
                    headers=self.headers,
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": texts,
                        "encoding_format": "float"
                    },
                    timeout=60
                )
                
                if response.status_code == 429:
                    # Rate limited — wait and retry
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 401:
                    raise ValueError(
                        "Mistral API key is invalid or expired. "
                        "Check your MISTRAL_API_KEY in the .env file."
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # Mistral returns embeddings sorted by index
                embeddings = [item["embedding"] for item in sorted(
                    data["data"], key=lambda x: x["index"]
                )]
                return embeddings
                
            except requests.exceptions.Timeout:
                last_error = f"Timeout on attempt {attempt + 1}"
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Embedding API failed after {MAX_RETRIES} attempts: {last_error}")
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string. Convenience wrapper."""
        embeddings = self.embed_texts([query])
        return embeddings[0]
    
    def test_connection(self) -> bool:
        """Test that the API key is valid and the service is reachable."""
        try:
            result = self.embed_texts(["test"])
            return len(result) > 0 and len(result[0]) == EMBEDDING_DIMENSIONS
        except Exception:
            return False


class VectorStore:
    """
    In-memory vector store using NumPy.
    
    Stores embeddings as a pre-normalized (N x D) float32 matrix.
    Cosine similarity is computed as a single matrix-vector multiplication.
    
    This replaces external vector databases (Pinecone, Qdrant, Chroma) with
    pure NumPy — demonstrating the underlying retrieval math directly.
    """
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._embeddings_matrix: Optional[np.ndarray] = None  # Shape: (N, 1024)
        self._doc_index: Dict[str, List[int]] = {}            # doc_id → chunk indices
    
    @property
    def size(self) -> int:
        return len(self.chunks)
    
    @property
    def document_ids(self) -> List[str]:
        return list(self._doc_index.keys())
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add embedded chunks to the vector store.
        Rebuilds the embedding matrix after each batch addition.
        
        Args:
            chunks: DocumentChunk objects with populated embedding fields
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding. "
                                "Embed chunks before adding to the store.")
            
            chunk_index = len(self.chunks)
            self.chunks.append(chunk)
            
            doc_id = chunk.metadata.doc_id
            if doc_id not in self._doc_index:
                self._doc_index[doc_id] = []
            self._doc_index[doc_id].append(chunk_index)
        
        # Rebuild the embedding matrix with L2 normalization
        self._rebuild_matrix()
    
    def _rebuild_matrix(self) -> None:
        """Rebuild and re-normalize the full embedding matrix."""
        if not self.chunks:
            self._embeddings_matrix = None
            return
        
        raw_matrix = np.array(
            [chunk.embedding for chunk in self.chunks],
            dtype=np.float32
        )
        
        # L2 normalize each row so cosine similarity = dot product
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)  # Avoid division by zero
        self._embeddings_matrix = raw_matrix / norms
    
    def cosine_similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find the top_k most similar chunks to the query embedding.
        Uses dot product on L2-normalized vectors (equivalent to cosine similarity).
        
        Args:
            query_embedding: The query's embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_index, similarity_score) tuples, sorted by score descending
        """
        if self._embeddings_matrix is None or self.size == 0:
            return []
        
        # Normalize query vector
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm
        
        # Compute all cosine similarities in one matrix multiplication
        # Shape: (N,) — one similarity score per chunk
        similarities = self._embeddings_matrix @ query_vec
        
        # Get top_k indices sorted by similarity (descending)
        top_k = min(top_k, self.size)
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_chunks_by_doc(self, doc_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document."""
        indices = self._doc_index.get(doc_id, [])
        return [self.chunks[i] for i in indices]
    
    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks for a document. Returns number of chunks removed."""
        if doc_id not in self._doc_index:
            return 0
        
        indices_to_remove = set(self._doc_index[doc_id])
        removed_count = len(indices_to_remove)
        
        self.chunks = [c for i, c in enumerate(self.chunks) if i not in indices_to_remove]
        del self._doc_index[doc_id]
        
        # Rebuild index mappings
        self._doc_index = {}
        for new_idx, chunk in enumerate(self.chunks):
            doc_id_key = chunk.metadata.doc_id
            if doc_id_key not in self._doc_index:
                self._doc_index[doc_id_key] = []
            self._doc_index[doc_id_key].append(new_idx)
        
        self._rebuild_matrix()
        return removed_count
    
    def clear(self) -> None:
        """Remove all chunks from the store."""
        self.chunks = []
        self._embeddings_matrix = None
        self._doc_index = {}


# ── Module-level singleton ────────────────────────────────────────────────────
# One shared vector store instance across the entire application lifecycle
_vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """Return the application-wide vector store singleton."""
    return _vector_store


def get_embedder() -> MistralEmbedder:
    """Create a MistralEmbedder using the API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )
    return MistralEmbedder(api_key=api_key)


def embed_and_store_chunks(chunks: List[DocumentChunk]) -> None:
    """
    Convenience function: embed a list of chunks and add them to the vector store.
    
    Args:
        chunks: DocumentChunk objects without embeddings
    """
    embedder = get_embedder()
    texts = [chunk.text for chunk in chunks]
    
    embeddings = embedder.embed_texts(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    store = get_vector_store()
    store.add_chunks(chunks)
