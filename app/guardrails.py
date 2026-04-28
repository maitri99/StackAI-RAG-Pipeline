"""
Guardrails: Query Refusal Policies and Hallucination Detection

Design Decisions:
-----------------
1. PII DETECTION:
   We use regex patterns to detect common PII categories before the query
   reaches the LLM. This prevents:
   - Accidentally echoing PII back to the user
   - Storing PII in logs
   - Using PII as a search query (leaking it to embedding API)
   
   Categories detected: SSN, credit card, phone numbers, email addresses,
   date of birth patterns, medical record numbers.

2. MEDICAL/LEGAL DISCLAIMER:
   Queries that appear to be seeking personal medical or legal advice
   (rather than factual information retrieval) trigger a disclaimer.
   We detect these via keyword patterns.
   The system can still answer but prepends a mandatory disclaimer.

3. HALLUCINATION DETECTION:
   After generation, we scan the answer for specific factual claims
   (numbers, dates, names, percentages, proper nouns) and verify each
   against the retrieved context. Claims not found in the context
   are flagged as potential hallucinations.
   
   This is a lightweight heuristic — production systems would use
   a dedicated NLI (Natural Language Inference) model for entailment checking.

4. INSUFFICIENT EVIDENCE:
   If the maximum semantic similarity score across all retrieved chunks
   is below the threshold, we return a refusal rather than hallucinating
   an answer from the LLM's training data.
"""

import re
from typing import List, Tuple

from app.models import QueryIntent


# ── PII Patterns ──────────────────────────────────────────────────────────────
PII_PATTERNS = [
    # SSN: 123-45-6789 or 123456789
    (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', "Social Security Number"),
    # Credit card: 4 groups of 4 digits
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', "Credit Card Number"),
    # Phone numbers
    (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone Number"),
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email Address"),
    # Medical record numbers (common patterns)
    (r'\b(?:MRN|MR#?|Record\s*#?)\s*:?\s*\d+\b', "Medical Record Number"),
    # Date of birth patterns
    (r'\bDOB\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', "Date of Birth"),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "IP Address"),
]

# ── Medical/Legal Disclaimer Triggers ─────────────────────────────────────────
MEDICAL_DISCLAIMER_PATTERNS = [
    r'\b(?:diagnos|treatment|prescrib|medication|symptom|dosage|drug|therapy)\b.*\b(?:my|I have|I am|me)\b',
    r'\bam I\b.*\b(?:sick|ill|infected|positive|negative|at risk)\b',
    r'\bshould I\b.*\b(?:take|stop|start|use|avoid)\b',
    r'\bwhat\s+(?:should|can|do)\s+I\b.*\b(?:take|eat|do|avoid)\b',
    r'\bmy\s+(?:doctor|physician|nurse|symptoms|diagnosis|condition|results)\b',
    r'\b(?:legal\s+advice|sue|lawsuit|liability|malpractice)\b',
    r'\bam I\s+(?:liable|responsible|at fault|negligent)\b',
]

MEDICAL_DISCLAIMER_TEXT = (
    "⚠️ DISCLAIMER: This system provides information retrieval from documents only. "
    "It is not a substitute for professional medical or legal advice. "
    "Always consult a qualified healthcare provider or attorney for personal guidance."
)

# ── Conversational Patterns (no KB search needed) ────────────────────────────
CONVERSATIONAL_PATTERNS = [
    r'^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings)[!\s.,]*$',
    r'^(?:how are you|how\'s it going|what\'s up|what can you do)[?!.]*$',
    r'^(?:thanks?|thank you|thx|ty)[!.]*$',
    r'^(?:ok|okay|got it|understood|sure|alright|cool)[!.]*$',
    r'^(?:bye|goodbye|see you|later|quit|exit)[!.]*$',
    r'^(?:yes|no|maybe|probably)[!.]*$',
    r'^\?{1,3}$',  # Just question marks
]

# ── Out of Scope Patterns ─────────────────────────────────────────────────────
OUT_OF_SCOPE_PATTERNS = [
    r'\b(?:write\s+(?:code|script|program)|debug|compile|install)\b',
    r'\b(?:create\s+(?:image|picture|photo|art))\b',
    r'\b(?:play|song|music|movie|show|game)\b',
    r'\b(?:weather|stock\s+price|sports\s+score|news)\b',
]


def detect_pii(text: str) -> List[Tuple[str, str]]:
    """
    Detect PII in the query text.
    
    Returns:
        List of (matched_text, pii_type) tuples. Empty list if no PII found.
    """
    detections = []
    for pattern, pii_type in PII_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            detections.append((match, pii_type))
    return detections


def classify_intent(query: str) -> QueryIntent:
    """
    Classify the intent of a user query into one of the QueryIntent categories.
    
    Classification order (priority):
    1. PII detected → refuse immediately
    2. Conversational → respond directly without KB search
    3. Medical/legal personal advice → answer with disclaimer
    4. Out of scope → gentle refusal
    5. Default → knowledge search (trigger RAG pipeline)
    
    Args:
        query: The user's raw query text
        
    Returns:
        QueryIntent enum value
    """
    query_lower = query.lower().strip()
    
    # Priority 1: PII detection
    pii_hits = detect_pii(query)
    if pii_hits:
        return QueryIntent.PII_DETECTED
    
    # Priority 2: Conversational / small talk
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return QueryIntent.CONVERSATIONAL
    
    # Priority 3: Medical/legal personal advice
    for pattern in MEDICAL_DISCLAIMER_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return QueryIntent.MEDICAL_DISCLAIMER
    
    # Priority 4: Out of scope
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return QueryIntent.OUT_OF_SCOPE
    
    # Default: knowledge search
    return QueryIntent.KNOWLEDGE_SEARCH


def get_conversational_response(query: str) -> str:
    """Generate a simple conversational response without hitting the KB."""
    query_lower = query.lower().strip()
    
    if re.search(r'\b(?:hi|hello|hey|howdy|greetings)\b', query_lower):
        return ("Hello! I'm a document Q&A assistant. "
                "Upload PDF documents and I'll answer questions about their contents. "
                "What would you like to know?")
    
    if re.search(r'\bhow are you\b', query_lower):
        return "I'm ready to help! Ask me anything about the documents you've uploaded."
    
    if re.search(r'\b(?:thanks?|thank you)\b', query_lower):
        return "You're welcome! Feel free to ask if you have more questions."
    
    if re.search(r'\b(?:bye|goodbye)\b', query_lower):
        return "Goodbye! Come back anytime you have questions about your documents."
    
    if re.search(r'\bwhat can you do\b', query_lower):
        return ("I can answer questions about PDF documents you've uploaded. "
                "I use semantic search and keyword matching to find relevant passages, "
                "then generate answers with citations. "
                "Try uploading a document and asking a question!")
    
    return "I'm here to help with questions about your documents. What would you like to know?"


def get_pii_refusal(pii_hits: List[Tuple[str, str]]) -> str:
    """Generate a refusal message for queries containing PII."""
    pii_types = list(set(pii_type for _, pii_type in pii_hits))
    type_str = ", ".join(pii_types)
    return (
        f"⚠️ Your query appears to contain sensitive personal information ({type_str}). "
        "For your privacy and security, please remove any personal identifiers "
        "(Social Security numbers, credit card numbers, medical record numbers, etc.) "
        "from your query before submitting."
    )


def get_out_of_scope_response() -> str:
    """Generate a gentle out-of-scope refusal."""
    return (
        "I'm specialized for answering questions about uploaded documents. "
        "Your query appears to be outside that scope. "
        "Please ask questions related to the content of your uploaded PDFs."
    )


def check_insufficient_evidence(
    chunks: List,
    threshold: float
) -> bool:
    """
    Determine if retrieved chunks meet the minimum similarity threshold.
    Returns True if evidence is insufficient (max score < threshold).
    """
    if not chunks:
        return True
    
    max_score = max(
        (chunk.rerank_score or chunk.hybrid_score)
        for chunk in chunks
    )
    return max_score < threshold


def detect_hallucinations(
    answer: str,
    context_chunks: List[str]
) -> List[str]:
    """
    Scan the generated answer for factual claims not supported by retrieved context.
    
    Strategy:
    - Extract sentences containing specific factual claims (numbers, proper nouns,
      percentages, quoted terms)
    - Check each claim sentence against the full context
    - Flag sentences where key claim terms are absent from context
    
    This is a heuristic, not a guaranteed hallucination detector.
    
    Args:
        answer: The generated answer text
        context_chunks: List of retrieved chunk texts used for generation
        
    Returns:
        List of sentences that may contain unsupported claims
    """
    if not answer or not context_chunks:
        return []
    
    full_context = " ".join(context_chunks).lower()
    
    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    flagged = []
    
    # Patterns that indicate specific factual claims worth checking
    claim_patterns = [
        r'\b\d+(?:\.\d+)?%',                    # Percentages
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',        # Dollar amounts
        r'\b(?:19|20)\d{2}\b',                   # Years
        r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|kg|lb|mm|cm)\b',  # Medical measurements
        r'"[^"]{5,}"',                           # Quoted phrases
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Proper nouns (Title Case)
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Check if this sentence contains a specific factual claim
        has_claim = any(re.search(p, sentence) for p in claim_patterns)
        if not has_claim:
            continue
        
        # Extract the key claim terms from this sentence
        claim_terms = []
        
        # Numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
        claim_terms.extend(numbers)
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]{5,})"', sentence)
        claim_terms.extend([q.lower() for q in quoted])
        
        # Check if key terms appear in context
        terms_not_in_context = []
        for term in claim_terms:
            if len(term) > 2 and term.lower() not in full_context:
                terms_not_in_context.append(term)
        
        # Flag if more than half the claim terms are absent from context
        if claim_terms and len(terms_not_in_context) > len(claim_terms) * 0.5:
            flagged.append(sentence[:200] + ("..." if len(sentence) > 200 else ""))
    
    return flagged


def get_medical_disclaimer() -> str:
    """Return the medical/legal disclaimer text."""
    return MEDICAL_DISCLAIMER_TEXT
