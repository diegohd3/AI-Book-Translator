"""
Utilities for document format detection and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SUPPORTED_FORMATS = {"txt", "epub"}
AUTO_FORMAT = "auto"


def normalize_format_name(format_name: Optional[str]) -> Optional[str]:
    """Normalize optional format name to lowercase."""
    if format_name is None:
        return None
    value = format_name.strip().lower()
    if not value:
        return None
    return value


def detect_document_format(input_path: Path, explicit_format: Optional[str] = None) -> str:
    """
    Detect input document format.

    Priority:
    1. Explicit `--input-format` when provided and not `auto`
    2. Input file extension
    """
    normalized = normalize_format_name(explicit_format)
    if normalized and normalized != AUTO_FORMAT:
        if normalized not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported input format '{normalized}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        return normalized

    suffix = input_path.suffix.strip().lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".epub":
        return "epub"

    raise ValueError(
        f"Could not detect input format from extension '{input_path.suffix}'. "
        "Use --input-format txt|epub."
    )
