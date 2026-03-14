"""
Glossary loading, validation, matching, and hashing helpers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from normalization import sha256_hexdigest

logger = logging.getLogger(__name__)

_ALLOWED_POLICIES = {"preferred", "forced"}


class GlossaryValidationError(ValueError):
    """Raised when a glossary JSON file has an invalid structure."""


@dataclass(frozen=True)
class GlossaryTerm:
    """Single glossary mapping entry."""

    source: str
    target: str
    notes: Optional[str] = None
    policy: str = "preferred"
    case_sensitive: bool = False

    def as_canonical_dict(self) -> dict:
        """Return a deterministic dictionary representation."""
        payload = {
            "source": self.source.strip(),
            "target": self.target.strip(),
            "policy": self.policy.strip().lower(),
            "case_sensitive": bool(self.case_sensitive),
        }
        notes = (self.notes or "").strip()
        if notes:
            payload["notes"] = notes
        return payload


@dataclass(frozen=True)
class GlossaryMatch:
    """A glossary term that matched a chunk source text."""

    term: GlossaryTerm
    count: int
    positions: List[Tuple[int, int]]


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
                item["policy"],
                str(item["case_sensitive"]),
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
        with open(path, "r", encoding="utf-8-sig") as handle:
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
        policy = raw_term.get("policy", "preferred")
        case_sensitive = raw_term.get("case_sensitive", False)

        if not isinstance(source, str) or not source.strip():
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'source' value"
            )
        if not isinstance(target, str) or not target.strip():
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'target' value"
            )
        if notes is not None and not isinstance(notes, str):
            raise GlossaryValidationError(
                f"Term at index {idx} has an invalid 'notes' value"
            )
        if not isinstance(policy, str) or policy.strip().lower() not in _ALLOWED_POLICIES:
            raise GlossaryValidationError(
                f"Term at index {idx} has invalid policy. Allowed: forced, preferred"
            )
        if not isinstance(case_sensitive, bool):
            raise GlossaryValidationError(
                f"Term at index {idx} has invalid 'case_sensitive' value"
            )

        terms.append(
            GlossaryTerm(
                source=source.strip(),
                target=target.strip(),
                notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
                policy=policy.strip().lower(),
                case_sensitive=case_sensitive,
            )
        )

    return terms


def iter_glossary_pairs(
    terms: Iterable[GlossaryTerm],
) -> List[tuple[str, str, Optional[str], str, bool]]:
    """Return glossary entries in deterministic order for prompt injection."""
    return [
        (
            term.source,
            term.target,
            term.notes,
            term.policy,
            term.case_sensitive,
        )
        for term in sorted(
            terms,
            key=lambda item: (
                item.source.casefold(),
                item.target.casefold(),
                item.policy,
                str(item.case_sensitive),
                (item.notes or "").casefold(),
            ),
        )
    ]


def find_relevant_terms(text: str, terms: Sequence[GlossaryTerm]) -> List[GlossaryMatch]:
    """Return glossary terms that appear in the source text chunk."""
    matches: List[GlossaryMatch] = []
    for term in terms:
        positions = _find_term_positions(
            text=text,
            phrase=term.source,
            case_sensitive=term.case_sensitive,
        )
        if positions:
            matches.append(
                GlossaryMatch(term=term, count=len(positions), positions=positions)
            )

    matches.sort(
        key=lambda item: (
            item.term.source.casefold(),
            item.term.target.casefold(),
            item.term.policy,
        )
    )
    return matches


def find_used_glossary_terms(
    translated_text: str,
    relevant_matches: Sequence[GlossaryMatch],
) -> List[str]:
    """Return glossary target terms that appear in translated text."""
    used: List[str] = []
    for match in relevant_matches:
        positions = _find_term_positions(
            text=translated_text,
            phrase=match.term.target,
            case_sensitive=match.term.case_sensitive,
        )
        if positions:
            used.append(match.term.target)
    return sorted(set(used), key=lambda item: item.casefold())


def detect_missing_forced_terms(
    relevant_matches: Sequence[GlossaryMatch],
    used_targets: Sequence[str],
) -> List[GlossaryTerm]:
    """Return forced terms that were relevant but not present in translation output."""
    used_lookup = {item.casefold() for item in used_targets}
    missing: List[GlossaryTerm] = []

    for match in relevant_matches:
        if match.term.policy != "forced":
            continue
        if match.term.target.casefold() not in used_lookup:
            missing.append(match.term)

    return missing


def _find_term_positions(
    *,
    text: str,
    phrase: str,
    case_sensitive: bool,
) -> List[Tuple[int, int]]:
    if not text or not phrase:
        return []

    escaped = re.escape(phrase)
    left_boundary = r"(?<!\w)" if phrase[0].isalnum() else ""
    right_boundary = r"(?!\w)" if phrase[-1].isalnum() else ""
    pattern = f"{left_boundary}{escaped}{right_boundary}"

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(pattern, flags)
    return [(match.start(), match.end()) for match in compiled.finditer(text)]
