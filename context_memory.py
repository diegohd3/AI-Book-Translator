"""
Book-level context memory used across chunk translations.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from glossary import GlossaryMatch
from style_profile import StyleProfile


_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_PLACE_RE = re.compile(r"\b(?:in|at|from|to|into|toward)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


@dataclass
class ChunkContextEntry:
    """Per-chunk context data for short-term prompt snapshots."""

    chunk_index: int
    chunk_signal: str
    recurring_names: List[str]
    places: List[str]
    invented_terms: List[str]
    repeated_phrases: List[str]
    glossary_relevant: List[str]
    glossary_used: List[str]
    notes: List[str] = field(default_factory=list)


class BookContextMemory:
    """Track lightweight book-level memory across translated chunks."""

    def __init__(self, style_profile: StyleProfile):
        self.style_profile = style_profile

        self._name_counter: Counter[str] = Counter()
        self._place_counter: Counter[str] = Counter()
        self._invented_counter: Counter[str] = Counter()
        self._phrase_counter: Counter[str] = Counter()
        self._decision_map: Dict[str, Set[str]] = defaultdict(set)

        self._notes: List[str] = []
        self._history: List[ChunkContextEntry] = []

    def update_from_chunk(
        self,
        *,
        chunk_index: int,
        source_text: str,
        translated_text: str,
        chunk_signal: str,
        glossary_matches: Sequence[GlossaryMatch],
        glossary_used: Sequence[str],
        warnings: Sequence[str],
    ) -> None:
        """Ingest chunk results and update cumulative memory."""
        names = _extract_names(source_text)
        places = _extract_places(source_text)
        invented_terms = _extract_invented_terms(source_text)
        repeated_phrases = _extract_repeated_phrases(source_text)

        self._name_counter.update(names)
        self._place_counter.update(places)
        self._invented_counter.update(invented_terms)
        self._phrase_counter.update(repeated_phrases)

        used_set = {term.casefold() for term in glossary_used}
        relevant_sources: List[str] = []
        for match in glossary_matches:
            relevant_sources.append(match.term.source)
            if match.term.target.casefold() in used_set:
                self._decision_map[match.term.source].add(match.term.target)

        warning_items = [item.strip() for item in warnings if item and item.strip()]
        if warning_items:
            self._notes.extend(warning_items)

        self._history.append(
            ChunkContextEntry(
                chunk_index=chunk_index,
                chunk_signal=chunk_signal,
                recurring_names=sorted(set(names)),
                places=sorted(set(places)),
                invented_terms=sorted(set(invented_terms)),
                repeated_phrases=sorted(set(repeated_phrases)),
                glossary_relevant=sorted(set(relevant_sources)),
                glossary_used=sorted(set(glossary_used)),
                notes=warning_items,
            )
        )

        _ = translated_text

    def prompt_snapshot(self, context_window: int) -> Dict[str, object]:
        """Return compact context data for prompt injection."""
        entries = self._history[-context_window:]
        recent_notes = [note for entry in entries for note in entry.notes][-6:]

        recent_decisions: Dict[str, str] = {}
        for entry in entries:
            for source_term in entry.glossary_relevant:
                targets = sorted(self._decision_map.get(source_term, set()))
                if targets:
                    recent_decisions[source_term] = targets[-1]

        return {
            "recurring_character_names": self._top_recurring(self._name_counter),
            "places": self._top_recurring(self._place_counter),
            "invented_terms": self._top_recurring(self._invented_counter),
            "key_repeated_phrases": self._top_recurring(self._phrase_counter),
            "translation_decisions": recent_decisions,
            "translation_notes": recent_notes,
            "tone_style_reminders": [
                self.style_profile.translation_strategy,
                self.style_profile.author_voice_notes,
                self.style_profile.dialogue_style,
                self.style_profile.narrator_style,
            ],
        }

    def recurring_terms_summary(self) -> Dict[str, List[str]]:
        """Return recurring term lists for reporting."""
        return {
            "characters": self._top_recurring(self._name_counter),
            "places": self._top_recurring(self._place_counter),
            "invented_terms": self._top_recurring(self._invented_counter),
            "key_repeated_phrases": self._top_recurring(self._phrase_counter),
        }

    def consistency_risks(self) -> List[str]:
        """Return consistency risks detected from tracked decisions."""
        risks: List[str] = []
        for source_term, targets in sorted(self._decision_map.items()):
            if len(targets) > 1:
                risks.append(
                    f"Recurring term '{source_term}' mapped inconsistently: {', '.join(sorted(targets))}"
                )
        return risks

    @property
    def chunk_entries(self) -> List[ChunkContextEntry]:
        """Expose immutable-like chunk history for reporting."""
        return list(self._history)

    def _top_recurring(self, counter: Counter[str], limit: int = 10) -> List[str]:
        items = [term for term, count in counter.items() if count >= 2]
        items.sort(key=lambda item: (-counter[item], item.casefold()))
        return items[:limit]


def _extract_names(text: str) -> List[str]:
    return [candidate.strip() for candidate in _NAME_RE.findall(text or "")]


def _extract_places(text: str) -> List[str]:
    return [candidate.strip() for candidate in _PLACE_RE.findall(text or "")]


def _extract_invented_terms(text: str) -> List[str]:
    words = _WORD_RE.findall(text or "")
    invented: List[str] = []
    for word in words:
        if "-" in word or "'" in word:
            invented.append(word)
            continue
        if word and word[0].isupper() and any(ch.isupper() for ch in word[1:]):
            invented.append(word)
    return invented


def _extract_repeated_phrases(text: str) -> List[str]:
    words = [word.lower() for word in _WORD_RE.findall(text or "")]
    if len(words) < 4:
        return []

    phrases: Counter[str] = Counter()
    for size in (2, 3):
        for idx in range(0, len(words) - size + 1):
            phrase = " ".join(words[idx : idx + size])
            phrases[phrase] += 1

    repeated = [phrase for phrase, count in phrases.items() if count >= 2]
    repeated.sort()
    return repeated
