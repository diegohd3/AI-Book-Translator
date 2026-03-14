"""
Glossary loading, validation, and hashing helpers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from normalization import sha256_hexdigest

logger = logging.getLogger(__name__)


class GlossaryValidationError(ValueError):
    """Raised when a glossary JSON file has an invalid structure."""


@dataclass(frozen=True)
class GlossaryTerm:
    """Single glossary mapping entry."""

    source: str
    target: str
    notes: Optional[str] = None

    def as_canonical_dict(self) -> dict:
        """Return a deterministic dictionary representation."""
        payload = {
            "source": self.source.strip(),
            "target": self.target.strip(),
        }
        notes = (self.notes or "").strip()
        if notes:
            payload["notes"] = notes
        return payload


@dataclass
class Glossary:
    """Loaded glossary object used by prompt builder and cache keys."""

    terms: List[GlossaryTerm]
    source_path: Optional[Path] = None

    def is_empty(self) -> bool:
        """Return True when no glossary terms are available."""
        return len(self.terms) == 0

    def glossary_hash(self) -> str:
        """
        Compute a deterministic hash for cache key partitioning.

        Returns:
            SHA-256 hash string or a fixed sentinel when glossary is empty.
        """
        if self.is_empty():
            return "no_glossary"

        canonical_terms = sorted(
            (term.as_canonical_dict() for term in self.terms),
            key=lambda item: (
                item["source"].casefold(),
                item["target"].casefold(),
                item.get("notes", "").casefold(),
            ),
        )
        canonical_payload = {"terms": canonical_terms}
        encoded = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_hexdigest(encoded)


def load_glossary(glossary_path: Optional[Path]) -> Glossary:
    """
    Load and validate a glossary JSON file.

    Args:
        glossary_path: Path to glossary JSON. If None, returns an empty glossary.
    """
    if glossary_path is None:
        return Glossary(terms=[], source_path=None)

    path = Path(glossary_path)
    if not path.exists():
        raise FileNotFoundError(f"Glossary file does not exist: {path}")
    if not path.is_file():
        raise GlossaryValidationError(f"Glossary path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise GlossaryValidationError(
            f"Glossary JSON is invalid ({path}): {exc}"
        ) from exc

    terms = _validate_terms(data)
    logger.info("Loaded glossary with %s terms from %s", len(terms), path)
    return Glossary(terms=terms, source_path=path)


def _validate_terms(payload: object) -> List[GlossaryTerm]:
    """Validate glossary payload and return parsed terms."""
    if not isinstance(payload, dict):
        raise GlossaryValidationError("Glossary root must be a JSON object")

    raw_terms = payload.get("terms")
    if raw_terms is None:
        raise GlossaryValidationError("Glossary must include a 'terms' array")
    if not isinstance(raw_terms, list):
        raise GlossaryValidationError("'terms' must be a JSON array")

    terms: List[GlossaryTerm] = []
    for idx, raw_term in enumerate(raw_terms):
        if not isinstance(raw_term, dict):
            raise GlossaryValidationError(
                f"Term at index {idx} must be a JSON object"
            )

        source = raw_term.get("source")
        target = raw_term.get("target")
        notes = raw_term.get("notes")

        if not isinstance(source, str) or not source.strip():
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'source' value"
            )
        if not isinstance(target, str) or not target.strip():
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'target' value"
            )
        if notes is not None and (not isinstance(notes, str)):
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'notes' value"
            )

        terms.append(
            GlossaryTerm(
                source=source.strip(),
                target=target.strip(),
                notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
            )
        )

    return terms


def iter_glossary_pairs(terms: Iterable[GlossaryTerm]) -> List[tuple[str, str, Optional[str]]]:
    """Return glossary entries in deterministic order for prompt injection."""
    return [
        (term.source, term.target, term.notes)
        for term in sorted(
            terms,
            key=lambda item: (
                item.source.casefold(),
                item.target.casefold(),
                (item.notes or "").casefold(),
            ),
        )
    ]
