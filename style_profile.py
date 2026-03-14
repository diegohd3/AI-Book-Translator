"""
Style profile loading and validation for literary translation behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from normalization import sha256_hexdigest


class StyleProfileValidationError(ValueError):
    """Raised when a style profile JSON file has an invalid structure."""


_REQUIRED_TEXT_FIELDS = (
    "source_language",
    "target_language",
    "genre",
    "tone",
    "audience",
    "formality",
    "translation_strategy",
    "author_voice_notes",
    "dialogue_style",
    "narrator_style",
)

_REQUIRED_LIST_FIELDS = (
    "forbidden_patterns",
    "preferred_patterns",
)


@dataclass(frozen=True)
class StyleProfile:
    """Strict literary style profile used to guide translation prompts."""

    source_language: str
    target_language: str
    genre: str
    tone: str
    audience: str
    formality: str
    translation_strategy: str
    author_voice_notes: str
    dialogue_style: str
    narrator_style: str
    forbidden_patterns: List[str]
    preferred_patterns: List[str]
    source_path: Optional[Path] = None

    def as_canonical_dict(self) -> Dict[str, object]:
        """Return a deterministic dictionary representation."""
        return {
            "source_language": self.source_language.strip(),
            "target_language": self.target_language.strip(),
            "genre": self.genre.strip(),
            "tone": self.tone.strip(),
            "audience": self.audience.strip(),
            "formality": self.formality.strip(),
            "translation_strategy": self.translation_strategy.strip(),
            "author_voice_notes": self.author_voice_notes.strip(),
            "dialogue_style": self.dialogue_style.strip(),
            "narrator_style": self.narrator_style.strip(),
            "forbidden_patterns": [item.strip() for item in self.forbidden_patterns],
            "preferred_patterns": [item.strip() for item in self.preferred_patterns],
        }

    def profile_hash(self) -> str:
        """Compute a deterministic profile hash used for cache/report partitioning."""
        encoded = json.dumps(
            self.as_canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_hexdigest(encoded)

    def as_prompt_lines(self) -> List[str]:
        """Render profile constraints as compact prompt lines."""
        return [
            f"Genre: {self.genre}",
            f"Tone: {self.tone}",
            f"Audience: {self.audience}",
            f"Formality: {self.formality}",
            f"Strategy: {self.translation_strategy}",
            f"Author voice notes: {self.author_voice_notes}",
            f"Dialogue style: {self.dialogue_style}",
            f"Narrator style: {self.narrator_style}",
            "Forbidden patterns: " + ", ".join(self.forbidden_patterns),
            "Preferred patterns: " + ", ".join(self.preferred_patterns),
        ]


def default_style_profile(source_language: str = "en", target_language: str = "es") -> StyleProfile:
    """Return a safe default profile when no explicit file is provided."""
    return StyleProfile(
        source_language=source_language,
        target_language=target_language,
        genre="general literary",
        tone="faithful literary tone",
        audience="adult general readers",
        formality="neutral",
        translation_strategy="faithful literary translation with natural Spanish flow",
        author_voice_notes="preserve narrative rhythm, subtext, and emotional intent",
        dialogue_style="native, idiomatic Spanish dialogue with distinct character voices",
        narrator_style="fluid literary narrator voice, avoid robotic syntax",
        forbidden_patterns=[
            "literal word-by-word calques",
            "mechanical syntax that sounds translated",
        ],
        preferred_patterns=[
            "natural Spanish phrasing",
            "cohesive paragraph rhythm",
        ],
        source_path=None,
    )


def load_style_profile(style_profile_path: Optional[Path]) -> StyleProfile:
    """Load and validate a style profile JSON file."""
    if style_profile_path is None:
        raise StyleProfileValidationError("Style profile path is required")

    path = Path(style_profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Style profile file does not exist: {path}")
    if not path.is_file():
        raise StyleProfileValidationError(f"Style profile path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise StyleProfileValidationError(
            f"Style profile JSON is invalid ({path}): {exc}"
        ) from exc

    profile = _validate_style_profile_payload(payload)
    return StyleProfile(**profile, source_path=path)


def validate_style_profile_file(style_profile_path: Path) -> None:
    """Validate a style profile file and raise on failure."""
    _ = load_style_profile(style_profile_path)


def _validate_style_profile_payload(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise StyleProfileValidationError("Style profile root must be a JSON object")

    validated: Dict[str, object] = {}

    for key in _REQUIRED_TEXT_FIELDS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise StyleProfileValidationError(
                f"Style profile field '{key}' must be a non-empty string"
            )
        validated[key] = value.strip()

    for key in _REQUIRED_LIST_FIELDS:
        value = payload.get(key)
        if not isinstance(value, list) or not value:
            raise StyleProfileValidationError(
                f"Style profile field '{key}' must be a non-empty list"
            )
        cleaned_items: List[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise StyleProfileValidationError(
                    f"Style profile field '{key}' has invalid item at index {idx}"
                )
            cleaned_items.append(item.strip())
        validated[key] = cleaned_items

    return validated
