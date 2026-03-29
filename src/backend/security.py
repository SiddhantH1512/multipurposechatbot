"""
Security utilities for input sanitization, prompt injection detection,
PII redaction, and file upload validation.

This module provides defense-in-depth for user-facing inputs before they
reach the LLM or are stored in the database.
"""

import re
from typing import List, Tuple

from fastapi import HTTPException, status


# ── Prompt Injection Patterns ──
# Compiled for performance; case-insensitive matching.
# ── Prompt Injection Patterns ──
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|prior|previous|your\s+system\s+prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an|if)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|the)", re.IGNORECASE),
    re.compile(r"override\s+(your|all|the|safety)", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*system\s*\]", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"ignore\s+(above|prior|all)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|initial)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"forget\s+your\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"remove\s+all\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+previous", re.IGNORECASE),
]

# ── PII Patterns ──
# Each pattern maps a PII type label to a compiled regex.
PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone_us": re.compile(
        r"\b(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

# ── HTML / Script tag pattern ──
HTML_TAG_PATTERN = re.compile(r"<\s*/?\s*(script|style|iframe|object|embed|form|input|link)[^>]*>", re.IGNORECASE)
ALL_HTML_TAGS = re.compile(r"<[^>]+>")


def detect_prompt_injection(text: str) -> bool:
    """
    Check if the input text contains known prompt injection patterns.

    Args:
        text: The raw user input string.

    Returns:
        True if a prompt injection pattern is detected, False otherwise.
    """
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Scan text for PII patterns and replace matches with redaction placeholders.

    Args:
        text: The input text to scan for PII.

    Returns:
        A tuple of (redacted_text, list_of_pii_types_found).
        For example: ("My SSN is [REDACTED_SSN]", ["SSN"])
    """
    redacted = text
    found_types: List[str] = []

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            found_types.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

    return redacted, found_types


def _strip_dangerous_html(text: str) -> str:
    """
    Remove dangerous HTML/script tags from input text.

    Strips <script>, <style>, <iframe>, <object>, <embed>, <form>, <input>,
    and <link> tags entirely. Other HTML tags are left as-is since they're
    likely part of legitimate document content.

    Args:
        text: Raw input text.

    Returns:
        Text with dangerous HTML tags removed.
    """
    return HTML_TAG_PATTERN.sub("", text)


def sanitize_input(text: str) -> Tuple[str, List[str]]:
    """
    Full input sanitization pipeline:
        1. Strip dangerous HTML/script tags
        2. Detect prompt injection (raises ValueError)
        3. Redact PII
    """
    if not text or not isinstance(text, str):
        return "", []

    # Step 1: Strip dangerous HTML
    cleaned = _strip_dangerous_html(text)

    # Step 2: Check for prompt injection
    if detect_prompt_injection(cleaned):
        matched = [p.pattern for p in INJECTION_PATTERNS if p.search(cleaned)]
        raise ValueError(f"Prompt injection detected. Blocked: {matched[:3]}")

    # Step 3: Redact PII
    sanitized, pii_types = redact_pii(cleaned)

    if pii_types:
        print(f"[SECURITY] PII redacted: {pii_types}")   # optional audit log

    return sanitized, pii_types


def validate_file_upload(filename: str, content_type: str, file_bytes: bytes) -> None:
    """
    Validate an uploaded file for security and correctness.

    Checks:
        - File extension is .pdf
        - Content-Type header is application/pdf
        - File magic bytes match PDF signature (%PDF-)
        - File size does not exceed 50 MB

    Args:
        filename: Original filename from the upload.
        content_type: MIME type from the Content-Type header.
        file_bytes: Raw bytes of the uploaded file.

    Raises:
        HTTPException 400: If any validation check fails, with a descriptive detail.
    """
    from src.config import Config

    # Check extension
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: '{filename}'. Only .pdf files are allowed.",
        )

    # Check Content-Type
    if content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: '{content_type}'. Expected 'application/pdf'.",
        )

    # Check magic bytes — PDF files start with %PDF-
    if not file_bytes or not file_bytes[:5] == b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (magic bytes mismatch).",
        )

    # Check file size
    max_size = Config.MAX_UPLOAD_SIZE_BYTES
    if len(file_bytes) > max_size:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large: {size_mb:.1f} MB. Maximum allowed: {max_size // (1024 * 1024)} MB.",
        )
