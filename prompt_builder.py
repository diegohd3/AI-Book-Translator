"""
Prompt builder module for constructing translation requests.

Creates literary-aware prompts with optional style profile, glossary constraints,
and rolling context memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from glossary import GlossaryMatch, GlossaryTerm, iter_glossary_pairs
from models import TextChunk
from style_profile import StyleProfile, default_style_profile


class PromptBuilder:
    """Builds stage-aware translation prompts for literary text."""

    def __init__(
        self,
        source_language: str = "English",
        target_language: str = "Spanish",
        glossary_terms: Optional[Sequence[GlossaryTerm]] = None,
        style_profile: Optional[StyleProfile] = None,
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.glossary_terms = list(glossary_terms or [])
        self.style_profile = style_profile or default_style_profile(
            source_language=source_language,
            target_language=target_language,
        )

    def set_glossary_terms(self, glossary_terms: Optional[Sequence[GlossaryTerm]]) -> None:
        """Update glossary terms used for prompt construction."""
        self.glossary_terms = list(glossary_terms or [])

    def set_style_profile(self, style_profile: StyleProfile) -> None:
        """Update style profile used for prompt generation."""
        self.style_profile = style_profile

    def build_system_prompt(
        self,
        *,
        stage: str = "draft",
        chunk_signal: str = "narration",
        context_snapshot: Optional[Dict[str, object]] = None,
        relevant_matches: Optional[Sequence[GlossaryMatch]] = None,
    ) -> str:
        """Build system-level instructions for a translation stage."""
        lines = [
            (
                f"You are a senior literary translator from {self.source_language} "
                f"to {self.target_language}."
            ),
            "Preserve literary flow, author intent, emotion, and paragraph structure.",
            "Produce natural, idiomatic Spanish that does not sound machine-translated.",
            "Avoid literal mechanical phrasing and keep character voices differentiated.",
            (
                "If placeholder tokens like [[TB_SEG_000001_START]] or "
                "[[TB_SEG_000001_END]] appear, keep them unchanged and in the same order."
            ),
        ]

        lines.extend(self._build_style_block())
        lines.append(self._build_signal_instruction(chunk_signal))

        context_block = self._build_context_block(context_snapshot)
        if context_block:
            lines.append(context_block)

        glossary_block = self._build_glossary_block(relevant_matches)
        if glossary_block:
            lines.append(glossary_block)

        if stage == "refinement":
            lines.append(
                "Refinement stage: improve fluency/readability and literary tone "
                "without adding or changing meaning from the source text."
            )

        return "\n".join(lines)

    def build_translation_prompt(
        self,
        chunk: TextChunk,
        *,
        stage: str = "draft",
        chunk_signal: str = "narration",
        context_snapshot: Optional[Dict[str, object]] = None,
        relevant_matches: Optional[Sequence[GlossaryMatch]] = None,
        draft_text: Optional[str] = None,
    ) -> str:
        """Build stage-specific input prompt for translation."""
        lines: List[str] = []
        lines.append(
            f"Task: {self.source_language} -> {self.target_language} literary translation."
        )
        lines.append(self._build_signal_instruction(chunk_signal))
        lines.append(
            "Preserve marker placeholders (for example [[TB_SEG_000001_START]]) exactly."
        )

        context_block = self._build_context_block(context_snapshot)
        if context_block:
            lines.append(context_block)

        glossary_block = self._build_glossary_block(relevant_matches)
        if glossary_block:
            lines.append(glossary_block)

        if stage == "draft":
            lines.extend(
                [
                    "Translate faithfully with natural Spanish cadence.",
                    "Keep subtext and emotional tension intact.",
                    "Return only the translated text.",
                    "",
                    "SOURCE TEXT:",
                    chunk.original_text,
                    "",
                    "TRANSLATION:",
                ]
            )
            return "\n".join(lines)

        if not draft_text:
            raise ValueError("draft_text is required for refinement stage prompts")

        lines.extend(
            [
                "Refine the draft into polished literary Spanish.",
                "Do not invent details or alter the source meaning.",
                "Preserve paragraph boundaries and pacing.",
                "Return only the refined text.",
                "",
                "SOURCE TEXT:",
                chunk.original_text,
                "",
                "STAGE 1 DRAFT:",
                draft_text,
                "",
                "REFINED TRANSLATION:",
            ]
        )
        return "\n".join(lines)

    def build_payload(
        self,
        chunk: TextChunk,
        *,
        stage: str = "draft",
        chunk_signal: str = "narration",
        context_snapshot: Optional[Dict[str, object]] = None,
        relevant_matches: Optional[Sequence[GlossaryMatch]] = None,
        draft_text: Optional[str] = None,
        api_format: str = "openai",
    ) -> dict:
        """Build an API request payload for a translation service."""
        if api_format == "openai":
            return {
                "instructions": self.build_system_prompt(
                    stage=stage,
                    chunk_signal=chunk_signal,
                    context_snapshot=context_snapshot,
                    relevant_matches=relevant_matches,
                ),
                "input": self.build_translation_prompt(
                    chunk,
                    stage=stage,
                    chunk_signal=chunk_signal,
                    context_snapshot=context_snapshot,
                    relevant_matches=relevant_matches,
                    draft_text=draft_text,
                ),
            }

        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "text": chunk.original_text,
            "chunk_index": chunk.index,
            "stage": stage,
        }

    def _build_style_block(self) -> List[str]:
        lines = ["Style profile:"]
        lines.extend(f"- {item}" for item in self.style_profile.as_prompt_lines())
        return lines

    @staticmethod
    def _build_signal_instruction(chunk_signal: str) -> str:
        mapping = {
            "narration": "Focus on narrator rhythm and descriptive continuity.",
            "dialogue": "Prioritize natural spoken Spanish with distinct character voices.",
            "emphasis": "Preserve dramatic intensity and emotional force.",
            "mixed": "Balance dialogue naturalness with dramatic narrative intensity.",
        }
        return mapping.get(chunk_signal, mapping["narration"])

    def _build_glossary_block(
        self,
        relevant_matches: Optional[Sequence[GlossaryMatch]],
    ) -> str:
        terms = self.glossary_terms
        if relevant_matches is not None:
            terms = [match.term for match in relevant_matches]

        if not terms:
            return ""

        lines = ["Glossary instructions:"]
        for source, target, notes, policy, _ in iter_glossary_pairs(terms):
            if policy == "forced":
                line = f'- FORCE when present: "{source}" -> "{target}"'
            else:
                line = f'- Prefer when present: "{source}" -> "{target}"'
            if notes:
                line = f"{line} ({notes})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _build_context_block(context_snapshot: Optional[Dict[str, object]]) -> str:
        if not context_snapshot:
            return ""

        lines: List[str] = ["Book context memory:"]
        for key in (
            "recurring_character_names",
            "places",
            "invented_terms",
            "key_repeated_phrases",
        ):
            values = context_snapshot.get(key)
            if isinstance(values, list) and values:
                lines.append(f"- {key}: {', '.join(str(item) for item in values)}")

        decisions = context_snapshot.get("translation_decisions")
        if isinstance(decisions, dict) and decisions:
            rendered = ", ".join(f"{src}->{dst}" for src, dst in sorted(decisions.items()))
            lines.append(f"- prior decisions: {rendered}")

        notes = context_snapshot.get("translation_notes")
        if isinstance(notes, list) and notes:
            lines.append(f"- notes: {' | '.join(str(item) for item in notes)}")

        reminders = context_snapshot.get("tone_style_reminders")
        if isinstance(reminders, list) and reminders:
            lines.append(f"- style reminders: {' | '.join(str(item) for item in reminders)}")

        return "\n".join(lines)
