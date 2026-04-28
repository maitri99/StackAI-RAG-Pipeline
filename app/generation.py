"""
Query Processing and Answer Generation

Design Decisions:
-----------------
1. QUERY TRANSFORMATION:
   Raw user queries are often sub-optimal for retrieval because:
   - They're too short ("side effects?")
   - They use casual language ("what happens if...")
   - They miss relevant terminology from the documents
   
   We use Mistral to expand the query into a retrieval-optimized form
   that includes likely terminology from the domain. This is called
   "HyDE" (Hypothetical Document Embeddings) — we generate what an
   ideal answer would look like and embed that for retrieval.
   
   Benefit: retrieval query semantically matches the style of document content.

2. ANSWER GENERATION PROMPT:
   The system prompt enforces strict grounding:
   - "Answer ONLY from the provided context"
   - "If the context doesn't contain the answer, say so explicitly"
   - "Cite the source passage for every factual claim"
   
   This is grounded generation — the LLM cannot use knowledge from
   its training data, only from retrieved passages.

3. STRUCTURED OUTPUTS:
   For queries with specific intent patterns (list requests, comparison
   requests), we switch to a structured prompt that produces markdown
   tables or numbered lists rather than prose paragraphs.
   
   Intent patterns detected:
   - "list", "enumerate", "what are the" → bulleted list format
   - "compare", "difference between", "vs" → markdown table format
   - "steps", "how to", "process" → numbered steps format
   - Default → prose paragraph format

4. CITATION EXTRACTION:
   The LLM is instructed to use [SOURCE_N] markers. We post-process
   the answer to extract these and link them back to the actual chunks.
"""

import os
import re
import requests
from typing import List, Optional, Tuple

from app.models import (
    DocumentChunk, RetrievedChunk, Citation, QueryIntent
)


MISTRAL_API_BASE = "https://api.mistral.ai/v1"
GENERATION_MODEL = "mistral-small-latest"
QUERY_TRANSFORM_MODEL = "mistral-small-latest"


# ── Answer shape detection ────────────────────────────────────────────────────
LIST_PATTERNS = [
    r'\b(?:list|enumerate|what are|name all|give me all|show me all)\b',
    r'\bwhat\s+(?:are|were)\s+the\b',
]

TABLE_PATTERNS = [
    r'\b(?:compare|comparison|difference between|versus|vs\.?)\b',
    r'\bhow does\s+.+\s+differ\b',
]

STEPS_PATTERNS = [
    r'\bhow\s+(?:to|do|can|should)\b',
    r'\bwhat\s+(?:are\s+the\s+)?steps\b',
    r'\bprocess\s+(?:for|of|to)\b',
    r'\bprocedure\b',
]


def detect_answer_shape(query: str) -> str:
    """
    Detect what format the answer should be in based on query intent.
    
    Returns:
        'list', 'table', 'steps', or 'prose'
    """
    query_lower = query.lower()
    
    for pattern in TABLE_PATTERNS:
        if re.search(pattern, query_lower):
            return 'table'
    
    for pattern in STEPS_PATTERNS:
        if re.search(pattern, query_lower):
            return 'steps'
    
    for pattern in LIST_PATTERNS:
        if re.search(pattern, query_lower):
            return 'list'
    
    return 'prose'


def transform_query(query: str, api_key: str) -> str:
    """
    Transform a user query into a retrieval-optimized form using HyDE
    (Hypothetical Document Embedding).
    
    Generates a short hypothetical passage that would answer the query,
    then uses that for retrieval. The hypothetical passage uses domain
    terminology likely found in the actual documents.
    
    Args:
        query: Original user query
        api_key: Mistral API key
        
    Returns:
        Transformed query string (or original if transformation fails)
    """
    try:
        response = requests.post(
            f"{MISTRAL_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": QUERY_TRANSFORM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a search query optimizer. Given a user question, "
                            "write a 2-3 sentence passage that would ideally answer the question. "
                            "Use formal, document-style language with domain-specific terminology. "
                            "Do NOT add any preamble — output only the passage itself."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}"
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.2
            },
            timeout=20
        )
        response.raise_for_status()
        
        transformed = response.json()["choices"][0]["message"]["content"].strip()
        
        # Return combined: original query + transformed version
        # This ensures we don't lose original terms while gaining expanded context
        return f"{query} {transformed}"
        
    except Exception:
        # Fall back to original query if transformation fails
        return query


def _build_context_string(chunks: List[RetrievedChunk]) -> str:
    """Build a formatted context string from retrieved chunks with source markers."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[SOURCE_{i}] (File: {chunk.metadata.filename}, "
            f"Page {chunk.metadata.page_number})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(context_parts)


def _get_format_instruction(answer_shape: str) -> str:
    """Get format-specific instructions for the generation prompt."""
    if answer_shape == 'list':
        return (
            "Format your answer as a bulleted list. "
            "Each bullet should be a complete, self-contained point with a source citation."
        )
    elif answer_shape == 'table':
        return (
            "Format your answer as a markdown comparison table with clear column headers. "
            "Only include rows with information explicitly found in the sources."
        )
    elif answer_shape == 'steps':
        return (
            "Format your answer as numbered steps in sequential order. "
            "Each step should reference its source."
        )
    else:
        return (
            "Write a clear, concise prose answer in 2-4 paragraphs. "
            "Each factual claim must reference its source."
        )


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    api_key: str,
    intent: QueryIntent,
    add_disclaimer: bool = False
) -> Tuple[str, List[Citation]]:
    """
    Generate a grounded answer using retrieved context chunks.
    
    The LLM is strictly instructed to only use the provided context.
    Source markers ([SOURCE_N]) in the answer are post-processed into
    structured Citation objects.
    
    Args:
        query: The user's original query
        chunks: Retrieved and re-ranked chunks
        api_key: Mistral API key
        intent: Classified query intent
        add_disclaimer: Whether to prepend a medical/legal disclaimer
        
    Returns:
        Tuple of (answer_text, list_of_citations)
    """
    if not chunks:
        return (
            "I couldn't find any relevant information in the uploaded documents "
            "to answer your question. Please ensure relevant documents are uploaded, "
            "or try rephrasing your question.",
            []
        )
    
    context_string = _build_context_string(chunks)
    answer_shape = detect_answer_shape(query)
    format_instruction = _get_format_instruction(answer_shape)
    
    system_prompt = f"""You are a precise document Q&A assistant. Your task is to answer questions STRICTLY based on provided source documents.

CRITICAL RULES:
1. Answer ONLY from the provided context below. Do NOT use knowledge from your training data.
2. Every factual claim MUST end with a citation marker like [SOURCE_1], [SOURCE_2], etc.
3. If the context does not contain enough information to answer, respond with: "INSUFFICIENT_EVIDENCE: [explain what's missing]"
4. Do not make up facts, numbers, dates, or names not found in the sources.
5. If different sources contradict each other, note the contradiction explicitly.
6. {format_instruction}

CONTEXT:
{context_string}"""

    user_message = f"Question: {query}\n\nAnswer (with citations):"
    
    try:
        response = requests.post(
            f"{MISTRAL_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GENERATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 1000,
                "temperature": 0.1  # Low temperature for factual accuracy
            },
            timeout=60
        )
        response.raise_for_status()
        
        raw_answer = response.json()["choices"][0]["message"]["content"].strip()
        
        # Extract citations from the answer
        citations = _extract_citations(raw_answer, chunks)
        
        # Clean up the answer (keep [SOURCE_N] markers for UI display)
        clean_answer = raw_answer
        
        # Add disclaimer if needed
        if add_disclaimer:
            from app.guardrails import get_medical_disclaimer
            clean_answer = get_medical_disclaimer() + "\n\n" + clean_answer
        
        return clean_answer, citations
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error generating answer: {str(e)}"
        return error_msg, []


def _extract_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
    """
    Extract citation objects from the answer text.
    
    Finds [SOURCE_N] markers in the answer and links them back to
    the corresponding retrieved chunks.
    
    Args:
        answer: Generated answer text with [SOURCE_N] markers
        chunks: The retrieved chunks (1-indexed in the answer)
        
    Returns:
        List of Citation objects
    """
    citations = []
    cited_indices = set()
    
    # Find all [SOURCE_N] references in the answer
    source_refs = re.findall(r'\[SOURCE_(\d+)\]', answer)
    
    for ref_str in source_refs:
        ref_idx = int(ref_str) - 1  # Convert to 0-indexed
        
        if ref_idx in cited_indices:
            continue  # Avoid duplicate citations
        
        if 0 <= ref_idx < len(chunks):
            chunk = chunks[ref_idx]
            cited_indices.add(ref_idx)
            
            # Take the most relevant excerpt from the chunk
            excerpt = chunk.text[:300].strip()
            if len(chunk.text) > 300:
                excerpt += "..."
            
            citations.append(Citation(
                chunk_id=chunk.chunk_id,
                filename=chunk.metadata.filename,
                page_number=chunk.metadata.page_number,
                relevant_excerpt=excerpt,
                confidence=round(chunk.rerank_score or chunk.hybrid_score, 4)
            ))
    
    return citations


def generate_insufficient_evidence_response(
    query: str,
    best_score: float,
    threshold: float
) -> str:
    """
    Generate an informative "insufficient evidence" response.
    
    Rather than a generic refusal, we explain exactly why we can't answer
    and what the user can do about it.
    """
    return (
        f"**Insufficient Evidence**\n\n"
        f"The most relevant passage found in your documents has a similarity score "
        f"of {best_score:.2f}, which is below the required threshold of {threshold:.2f}.\n\n"
        f"This means the uploaded documents likely do not contain information directly "
        f"relevant to your question: *\"{query}\"*\n\n"
        f"**Suggestions:**\n"
        f"- Upload additional documents that cover this topic\n"
        f"- Rephrase your question using terminology from the documents\n"
        f"- Lower the similarity threshold in your query settings"
    )
