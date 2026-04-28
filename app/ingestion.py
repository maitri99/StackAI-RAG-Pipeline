"""
PDF Ingestion Pipeline — Text Extraction and Chunking

Design Decisions:
-----------------
1. TEXT EXTRACTION: pdfminer.six is used over PyPDF2 because it handles:
   - Complex layouts (multi-column, tables)
   - Font encoding edge cases
   - Better whitespace preservation
   We extract page-by-page to preserve page number metadata.

2. CHUNKING STRATEGY: Recursive character splitting with overlap.
   - Chunk size: 512 tokens (~2000 chars) — balances context richness vs. retrieval precision.
     Too large: retrieved chunks contain irrelevant text, diluting the signal.
     Too small: chunks lose surrounding context needed to answer questions.
   - Overlap: 10% (200 chars) — ensures sentences split across chunk boundaries
     are still retrievable from both sides. Critical for questions whose answers
     span paragraph breaks.
   - Splitting hierarchy: paragraph → sentence → word.
     We try to split on natural boundaries before hard-cutting mid-sentence.

3. CHUNK METADATA: Every chunk carries doc_id, filename, page number, and character
   offsets so the UI can show exactly where in the source document an answer came from.
"""

import hashlib
import re
import uuid
from io import BytesIO
from typing import List, Tuple, Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTPage 

from app.models import ChunkMetadata, DocumentChunk


# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1800        # characters (~450 tokens at ~4 chars/token)
CHUNK_OVERLAP = 200      # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 100     # discard chunks smaller than this (usually header artifacts)

# Split hierarchy — try these separators in order before hard-cutting
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _extract_text_by_page(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extract text from a PDF, returning a list of (page_number, text) tuples.
    Page numbers are 1-indexed.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        
    Returns:
        List of (page_number, page_text) tuples
    """
    pages = []
    try:
        pdf_stream = BytesIO(pdf_bytes)
        for page_num, page_layout in enumerate(extract_pages(pdf_stream), start=1):
            page_text_parts = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text()
                    if text.strip():
                        page_text_parts.append(text)
            
            page_text = "\n".join(page_text_parts)
            page_text = _clean_text(page_text)
            
            if page_text.strip():
                pages.append((page_num, page_text))
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    return pages


def _clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalize whitespace (collapse multiple spaces/newlines)
    - Remove null bytes and other control characters
    - Preserve paragraph breaks (double newlines)
    """
    # Remove null bytes and control chars (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize multiple spaces to single space (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize 3+ newlines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Recursively split text into chunks using a hierarchy of separators.
    Tries each separator in SPLIT_SEPARATORS order, preferring natural
    linguistic boundaries over hard cuts.
    
    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        overlap: Characters of overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []
    
    chunks = []
    
    # Find the best separator to split on
    best_separator = ""
    for sep in SPLIT_SEPARATORS:
        if sep in text:
            best_separator = sep
            break
    
    if best_separator == "":
        # Hard cut — no good separator found
        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]
        return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]
    
    # Split on separator and recombine into chunks
    splits = text.split(best_separator)
    current_chunk = ""
    
    for split in splits:
        candidate = current_chunk + best_separator + split if current_chunk else split
        
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk if it's long enough
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk)
            
            # If the single split itself is too large, recurse
            if len(split) > chunk_size:
                sub_chunks = _recursive_split(split, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                # Start new chunk, but include overlap from end of previous chunk
                if current_chunk and overlap > 0:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + best_separator + split
                else:
                    current_chunk = split
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk)
    
    return chunks


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    doc_id: Optional[str] = None
) -> List[DocumentChunk]:
    """
    Full ingestion pipeline for a single PDF:
    1. Extract text page-by-page
    2. Chunk each page's text with overlap
    3. Attach metadata to each chunk
    4. Return list of DocumentChunk objects (without embeddings — those are added separately)
    
    Args:
        pdf_bytes: Raw PDF bytes
        filename: Original filename (used in citations)
        doc_id: Optional doc ID; generated from filename hash if not provided
        
    Returns:
        List of DocumentChunk objects ready for embedding
    """
    if doc_id is None:
        # Deterministic ID from filename + content hash
        content_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        doc_id = f"{filename.replace('.pdf', '').replace(' ', '_')}_{content_hash}"
    
    pages = _extract_text_by_page(pdf_bytes)
    
    if not pages:
        raise ValueError(f"No text content extracted from {filename}. "
                        "The PDF may be image-based (scanned) and requires OCR.")
    
    all_chunks: List[DocumentChunk] = []
    global_chunk_index = 0
    
    for page_num, page_text in pages:
        page_chunks = _recursive_split(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for local_idx, chunk_text in enumerate(page_chunks):
            # Approximate character offsets (exact offsets would require tracking positions)
            char_start = page_text.find(chunk_text[:50]) if chunk_text else 0
            char_end = char_start + len(chunk_text)
            
            chunk_id = str(uuid.uuid4())
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=global_chunk_index,
                    total_chunks=-1,  # Updated after processing all pages
                    char_start=max(0, char_start),
                    char_end=char_end
                )
            )
            all_chunks.append(chunk)
            global_chunk_index += 1
    
    # Update total_chunks now that we know the full count
    for chunk in all_chunks:
        chunk.metadata.total_chunks = len(all_chunks)
    
    return all_chunks


