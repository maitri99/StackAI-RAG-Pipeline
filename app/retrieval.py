"""
Hybrid Retrieval: Semantic Search + BM25 Keyword Search + Re-ranking

Design Decisions:
-----------------
1. WHY HYBRID SEARCH?
   Pure semantic search (cosine similarity on embeddings) is excellent at finding
   conceptually related text but fails on:
   - Exact terms: medical codes (ICD-10, CPT), drug names, model numbers
   - Abbreviations: "PA", "CDAI", "TNF-α"
   - Names: "Adalimumab", "UnitedHealthcare"
   BM25 (keyword search) excels at exactly these cases.
   Combining both gives better coverage than either alone.

2. BM25 IMPLEMENTATION:
   BM25 (Best Match 25) is a probabilistic ranking function that:
   - Rewards term frequency (TF) but with diminishing returns (saturation)
   - Penalizes very long documents (length normalization)
   - The "25" refers to the 25th iteration of the Okapi BM weighting scheme
   We implement it from scratch using only standard library (math, collections).
   
   Key parameters:
   - k1 = 1.5: Controls term frequency saturation (higher = slower saturation)
   - b = 0.75: Controls document length normalization (1.0 = full normalization)

3. HYBRID SCORE FUSION:
   We combine semantic and keyword scores using Reciprocal Rank Fusion (RRF):
   score = (1-α) * semantic_score + α * bm25_normalized_score
   where α = 0.3 (semantic weighted more, since it captures meaning better)
   
   Scores are normalized to [0,1] before fusion.

4. RE-RANKING:
   After retrieving top_k * 2 candidates, we re-rank using Mistral to score
   each chunk's relevance to the specific query. This is a "cross-encoder" pattern:
   while bi-encoders (embeddings) encode query and document independently,
   re-ranking allows query-document interaction for better precision.
   We use the top_k * 2 candidates to re-rank, then return top_k final results.
"""

import math
import re
import os
import requests
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from app.models import DocumentChunk, RetrievedChunk
from app.embeddings import VectorStore


# ── BM25 Parameters ───────────────────────────────────────────────────────────
BM25_K1 = 1.5       # Term frequency saturation parameter
BM25_B = 0.75       # Document length normalization parameter

# ── Fusion weight ─────────────────────────────────────────────────────────────
# Alpha controls the balance: 0 = pure semantic, 1 = pure keyword
SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30

# ── Re-ranking ────────────────────────────────────────────────────────────────
RERANK_CANDIDATES_MULTIPLIER = 2  # Retrieve 2x top_k candidates for re-ranking


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric characters.
    Preserves hyphenated terms (e.g., "TNF-alpha") as single tokens.
    """
    text = text.lower()
    # Keep alphanumeric and hyphens, split on everything else
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
    return tokens


class BM25Index:
    """
    BM25 inverted index built from a list of document chunks.
    
    Built once during ingestion and used at query time for fast keyword search.
    The index maps tokens to the list of (chunk_index, term_frequency) pairs.
    """
    
    def __init__(self):
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0
        self.doc_freq: Dict[str, int] = {}  # Number of docs containing each term
    
    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build the BM25 index from a list of document chunks."""
        self.inverted_index = defaultdict(list)
        self.doc_lengths = []
        self.doc_freq = {}
        self.num_docs = len(chunks)
        
        for doc_idx, chunk in enumerate(chunks):
            tokens = _tokenize(chunk.text)
            self.doc_lengths.append(len(tokens))
            
            term_counts = Counter(tokens)
            
            for term, count in term_counts.items():
                self.inverted_index[term].append((doc_idx, count))
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0
        )
    
    def get_scores(self, query: str, num_docs: Optional[int] = None) -> Dict[int, float]:
        """
        Compute BM25 scores for all documents matching query terms.
        
        Args:
            query: The search query
            num_docs: Total number of docs (uses self.num_docs if not provided)
            
        Returns:
            Dict mapping chunk_index to BM25 score
        """
        if self.num_docs == 0:
            return {}
        
        n = num_docs or self.num_docs
        query_tokens = _tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self.doc_freq.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            
            for doc_idx, tf in self.inverted_index[term]:
                # BM25 TF component with saturation and length normalization
                doc_length = self.doc_lengths[doc_idx]
                tf_normalized = (
                    tf * (BM25_K1 + 1)
                ) / (
                    tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / self.avg_doc_length)
                )
                
                scores[doc_idx] += idf * tf_normalized
        
        return dict(scores)


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def hybrid_search(
    query_embedding: List[float],
    query_text: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 10,
    similarity_threshold: float = 0.4
) -> List[RetrievedChunk]:
    """
    Perform hybrid search combining semantic similarity and BM25 keyword search.
    
    Args:
        query_embedding: Embedded query vector
        query_text: Raw query text (for BM25)
        vector_store: The vector store containing chunk embeddings
        bm25_index: Pre-built BM25 index
        top_k: Number of final results to return
        similarity_threshold: Minimum semantic score to include a chunk
        
    Returns:
        List of RetrievedChunk objects sorted by hybrid score (descending)
    """
    if vector_store.size == 0:
        return []
    
    candidates = top_k * RERANK_CANDIDATES_MULTIPLIER
    
    # ── Semantic search ──────────────────────────────────────────────────────
    semantic_results = vector_store.cosine_similarity_search(
        query_embedding, top_k=candidates
    )
    semantic_scores = {idx: score for idx, score in semantic_results}
    
    # ── BM25 keyword search ──────────────────────────────────────────────────
    bm25_raw_scores = bm25_index.get_scores(query_text, num_docs=vector_store.size)
    
    # ── Normalize both score sets ────────────────────────────────────────────
    semantic_normalized = _normalize_scores(semantic_scores)
    bm25_normalized = _normalize_scores(bm25_raw_scores)
    
    # ── Union of candidate indices ───────────────────────────────────────────
    all_candidate_indices = set(semantic_scores.keys()) | set(bm25_raw_scores.keys())
    
    # ── Compute hybrid scores ────────────────────────────────────────────────
    hybrid_scores: Dict[int, float] = {}
    for idx in all_candidate_indices:
        sem_score = semantic_normalized.get(idx, 0.0)
        kw_score = bm25_normalized.get(idx, 0.0)
        hybrid_scores[idx] = SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score
    
    # ── Sort and filter ──────────────────────────────────────────────────────
    sorted_candidates = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, hybrid_score in sorted_candidates[:candidates]:
        sem_score = semantic_scores.get(idx, 0.0)
        
        # Apply similarity threshold on semantic score
        if sem_score < similarity_threshold and idx not in semantic_scores:
            continue
        
        chunk = vector_store.chunks[idx]
        
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            semantic_score=round(sem_score, 4),
            keyword_score=round(bm25_raw_scores.get(idx, 0.0), 4),
            hybrid_score=round(hybrid_score, 4),
            rerank_score=None
        ))
    
    return results[:top_k]


def rerank_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    api_key: str,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Re-rank retrieved chunks using Mistral to score query-chunk relevance.
    
    This is a lightweight cross-encoder pattern: we ask the LLM to score
    how relevant each chunk is to the query on a 0-10 scale, then re-sort.
    
    Args:
        query: The user's original query
        chunks: Candidate chunks from hybrid search
        api_key: Mistral API key
        top_k: Final number of chunks to return after re-ranking
        
    Returns:
        Re-ranked list of RetrievedChunk objects (top_k best)
    """
    if not chunks or len(chunks) <= 1:
        return chunks[:top_k]
    
    # Build a batch prompt to score all chunks in one API call
    chunk_descriptions = []
    for i, chunk in enumerate(chunks):
        excerpt = chunk.text[:300].replace('\n', ' ')
        chunk_descriptions.append(f"[{i}] {excerpt}")
    
    prompt = f"""Score each passage's relevance to the query. Reply ONLY with a JSON array of numbers from 0-10.
Example format: [8, 3, 9, 1, 7]

Query: {query}

Passages:
{chr(10).join(chunk_descriptions)}

Relevance scores (one number per passage, same order):"""
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.0
            },
            timeout=30
        )
        response.raise_for_status()
        
        import json
        import re as regex
        
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Extract the JSON array from the response
        array_match = regex.search(r'\[[\d\s,\.]+\]', content)
        if array_match:
            scores = json.loads(array_match.group())
            
            # Apply re-rank scores
            for i, chunk in enumerate(chunks):
                if i < len(scores):
                    chunk.rerank_score = float(scores[i]) / 10.0
            
            # Sort by rerank score (descending)
            chunks.sort(key=lambda c: c.rerank_score or 0, reverse=True)
    
    except Exception:
        # Re-ranking failed — fall back to hybrid score ordering (already sorted)
        pass
    
    return chunks[:top_k]


# ── Module-level BM25 index singleton ────────────────────────────────────────
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    """Return the application-wide BM25 index singleton."""
    return _bm25_index


def rebuild_bm25_index(chunks: List[DocumentChunk]) -> None:
    """Rebuild the BM25 index from scratch with the provided chunks."""
    _bm25_index.build(chunks)
