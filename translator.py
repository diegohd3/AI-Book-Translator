"""
Translation module with pluggable translator backends.

Provides an abstraction layer for different translation services
and implementations for mock and real translators.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

from glossary import GlossaryTerm
from models import TextChunk, TranslatedChunk
from prompt_builder import PromptBuilder
from style_profile import StyleProfile

logger = logging.getLogger(__name__)

_MOCK_TRANSLATION_TERMS = (
    ("the", "el/la"),
    ("is", "es"),
    ("and", "y"),
    ("a", "un/una"),
    ("to", "a"),
    ("of", "de"),
    ("in", "en"),
    ("that", "ese/esa"),
    ("it", "lo/la"),
    ("for", "para"),
)
_MOCK_TRANSLATION_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE), target)
    for source, target in _MOCK_TRANSLATION_TERMS
)


class BaseTranslator(ABC):
    """Abstract base class for translation implementations."""

    @abstractmethod
    def translate(
        self,
        chunk: TextChunk,
        *,
        prompt_metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        """Translate a source chunk (stage 1 draft)."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return translator display name."""
        raise NotImplementedError

    def refine(
        self,
        chunk: TextChunk,
        draft_text: str,
        *,
        prompt_metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        """Refine a stage 1 translation. Default behavior is no-op."""
        _ = prompt_metadata
        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=draft_text,
            word_count=chunk.word_count,
            translator_used=f"{self.get_name()}[refinement-no-op]",
            metadata={"refinement": "no-op"},
        )

    def configure_glossary(self, glossary_terms: Sequence[GlossaryTerm]) -> None:
        """Optional hook for glossary-aware translators."""
        _ = glossary_terms

    def configure_style_profile(self, style_profile: StyleProfile) -> None:
        """Optional hook for style-aware translators."""
        _ = style_profile

    def supports_refinement(self) -> bool:
        """Return True when translator can run a real stage-2 refinement call."""
        return False


class MockTranslator(BaseTranslator):
    """Mock translator for testing and development."""

    def __init__(self, target_language: str = "es"):
        self.target_language = target_language

    def translate(
        self,
        chunk: TextChunk,
        *,
        prompt_metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        logger.info("MockTranslator: translating chunk %s", chunk.index)
        translated_text = self._mock_translate(chunk.original_text)

        metadata = _sanitize_prompt_metadata(prompt_metadata)
        metadata["stage"] = "draft"

        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=translated_text,
            word_count=chunk.word_count,
            translator_used="MockTranslator",
            metadata=metadata,
        )

    def _mock_translate(self, text: str) -> str:
        translated = text
        for pattern, replacement in _MOCK_TRANSLATION_PATTERNS:
            translated = pattern.sub(replacement, translated)

        return f"[ES MOCK TRANSLATION]\n{translated}"

    def get_name(self) -> str:
        return "MockTranslator"


class OpenAITranslator(BaseTranslator):
    """OpenAI-based translator implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",
        source_language: str = "English",
        target_language: str = "Spanish",
        glossary_terms: Optional[Sequence[GlossaryTerm]] = None,
        style_profile: Optional[StyleProfile] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model.strip()
        self.source_language = self._normalize_language(source_language)
        self.target_language = self._normalize_language(target_language)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY."
            )
        if not self.model:
            raise ValueError("OpenAI model is required")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.client = OpenAI(api_key=self.api_key)
        self.prompt_builder = PromptBuilder(
            source_language=self.source_language,
            target_language=self.target_language,
            glossary_terms=glossary_terms,
            style_profile=style_profile,
        )

    def translate(
        self,
        chunk: TextChunk,
        *,
        prompt_metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        return self._translate_stage(
            chunk,
            stage="draft",
            prompt_metadata=prompt_metadata,
            draft_text=None,
        )

    def refine(
        self,
        chunk: TextChunk,
        draft_text: str,
        *,
        prompt_metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        return self._translate_stage(
            chunk,
            stage="refinement",
            prompt_metadata=prompt_metadata,
            draft_text=draft_text,
        )

    def supports_refinement(self) -> bool:
        return True

    def configure_glossary(self, glossary_terms: Sequence[GlossaryTerm]) -> None:
        self.prompt_builder.set_glossary_terms(glossary_terms)

    def configure_style_profile(self, style_profile: StyleProfile) -> None:
        self.prompt_builder.set_style_profile(style_profile)

    def _translate_stage(
        self,
        chunk: TextChunk,
        *,
        stage: str,
        prompt_metadata: Optional[dict],
        draft_text: Optional[str],
    ) -> TranslatedChunk:
        metadata = dict(prompt_metadata or {})
        chunk_signal = str(metadata.get("chunk_signal", "narration"))
        context_snapshot = metadata.get("context_snapshot")
        relevant_matches = metadata.get("relevant_glossary_terms")

        logger.info(
            "OpenAITranslator (%s): %s chunk %s",
            self.model,
            stage,
            chunk.index,
        )

        try:
            payload = self.prompt_builder.build_payload(
                chunk,
                stage=stage,
                chunk_signal=chunk_signal,
                context_snapshot=context_snapshot if isinstance(context_snapshot, dict) else None,
                relevant_matches=relevant_matches,
                draft_text=draft_text,
                api_format="openai",
            )

            response = self.client.responses.create(
                model=self.model,
                instructions=payload["instructions"],
                input=payload["input"],
            )

            translated_text = self._extract_output_text(response)
            usage = self._extract_usage_metadata(response)

            if not translated_text:
                raise ValueError("OpenAI API returned an empty translation response")

            response_metadata: Dict[str, object] = {
                "stage": stage,
                "usage": usage,
            }
            response_metadata.update(_sanitize_prompt_metadata(metadata))

            return TranslatedChunk(
                index=chunk.index,
                original_text=chunk.original_text,
                translated_text=translated_text,
                word_count=chunk.word_count,
                translator_used=f"OpenAITranslator({self.model})",
                estimated_tokens=usage.get("total_tokens") if usage else None,
                metadata=response_metadata,
            )

        except Exception as exc:
            logger.error("OpenAI %s failed for chunk %s: %s", stage, chunk.index, exc)
            raise RuntimeError(
                f"OpenAI {stage} failed for chunk {chunk.index}: {exc}"
            ) from exc

    @staticmethod
    def _extract_output_text(response) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = getattr(response, "output", [])
        text_parts = []
        for item in output_items:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_usage_metadata(response) -> Dict[str, int]:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return {}

        usage: Dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage_obj, key, None)
            if isinstance(value, int):
                usage[key] = value
        return usage

    @staticmethod
    def _normalize_language(language: str) -> str:
        mapping = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        key = language.strip().lower()
        return mapping.get(key, language)

    def get_name(self) -> str:
        return f"OpenAITranslator({self.model})"


class TranslatorFactory:
    """Factory for creating translator instances."""

    _translators = {
        "mock": MockTranslator,
        "openai": OpenAITranslator,
    }

    @classmethod
    def create(
        cls,
        translator_type: str = "mock",
        source_language: str = "English",
        target_language: str = "Spanish",
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",
        glossary_terms: Optional[Sequence[GlossaryTerm]] = None,
        style_profile: Optional[StyleProfile] = None,
    ) -> BaseTranslator:
        if translator_type not in cls._translators:
            raise ValueError(
                f"Unknown translator type: {translator_type}. "
                f"Supported types: {', '.join(cls._translators.keys())}"
            )

        translator_class = cls._translators[translator_type]

        if translator_type == "mock":
            return translator_class(target_language=target_language)
        if translator_type == "openai":
            return translator_class(
                api_key=api_key,
                model=model,
                source_language=source_language,
                target_language=target_language,
                glossary_terms=glossary_terms,
                style_profile=style_profile,
            )

        raise ValueError(f"Cannot create translator: {translator_type}")

    @classmethod
    def register(cls, translator_type: str, translator_class: type) -> None:
        cls._translators[translator_type] = translator_class
        logger.info("Registered translator type: %s", translator_type)


def _sanitize_prompt_metadata(prompt_metadata: Optional[dict]) -> Dict[str, object]:
    """Return prompt metadata stripped to JSON-safe observability fields."""
    if not prompt_metadata:
        return {}

    safe: Dict[str, object] = {}
    chunk_signal = prompt_metadata.get("chunk_signal")
    if isinstance(chunk_signal, str):
        safe["chunk_signal"] = chunk_signal

    context_snapshot = prompt_metadata.get("context_snapshot")
    if isinstance(context_snapshot, dict):
        safe["context_snapshot"] = context_snapshot

    relevant_terms = prompt_metadata.get("relevant_glossary_terms")
    if isinstance(relevant_terms, list):
        rendered_terms = []
        for item in relevant_terms:
            term = getattr(item, "term", None)
            source = getattr(term, "source", None) if term is not None else None
            target = getattr(term, "target", None) if term is not None else None
            if isinstance(source, str) and isinstance(target, str):
                rendered_terms.append(f"{source} -> {target}")
        if rendered_terms:
            safe["relevant_glossary_terms"] = rendered_terms

    return safe
