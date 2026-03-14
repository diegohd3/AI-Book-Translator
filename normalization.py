"""
Utilities for deterministic text normalization and key generation.

This module is shared by cache and translation memory lookups.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text_for_lookup(text: str) -> str:
    """
    Normalize text for exact-match storage and retrieval.

    The normalization is intentionally conservative: it preserves case and
    punctuation while normalizing Unicode and whitespace.
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def sha256_hexdigest(value: str) -> str:
    """Return a SHA-256 hex digest for a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_deterministic_key(*parts: str) -> str:
    """
    Build a deterministic SHA-256 key from multiple string parts.

    Uses a stable delimiter unlikely to appear naturally in text.
    """
    joined = "\x1f".join(parts)
    return sha256_hexdigest(joined)
