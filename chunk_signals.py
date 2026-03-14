"""
Chunk signal detection helpers for dialogue-aware prompting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_DIALOGUE_LINE_RE = re.compile(r"^\s*([\"'\u201c\u201d\u2018\u2019\u2014\-])")
_ALL_CAPS_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{3,}\b")


@dataclass(frozen=True)
class ChunkSignal:
    """Classification scores used by prompt generation."""

    label: str
    dialogue_score: float
    emphasis_score: float


def analyze_chunk_signal(text: str) -> ChunkSignal:
    """Classify a text chunk as narration, dialogue, emphasis, or mixed."""
    content = text or ""
    lines = [line for line in content.splitlines() if line.strip()]
    quote_count = sum(content.count(symbol) for symbol in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019"))
    dialogue_lines = sum(1 for line in lines if _DIALOGUE_LINE_RE.match(line) is not None)

    line_count = max(1, len(lines))
    dialogue_score = min(1.0, (quote_count / 6.0) + (dialogue_lines / line_count))

    exclamations = content.count("!") + content.count("\u00a1")
    questions = content.count("?") + content.count("\u00bf")
    caps_words = len(_ALL_CAPS_RE.findall(content))
    emphasis_score = min(1.0, (exclamations + questions) / 8.0 + caps_words / 6.0)

    label = "narration"
    if dialogue_score >= 0.45 and emphasis_score >= 0.35:
        label = "mixed"
    elif dialogue_score >= 0.45:
        label = "dialogue"
    elif emphasis_score >= 0.35:
        label = "emphasis"

    return ChunkSignal(
        label=label,
        dialogue_score=round(dialogue_score, 4),
        emphasis_score=round(emphasis_score, 4),
    )
