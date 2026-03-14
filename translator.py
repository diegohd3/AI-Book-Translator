"""
Translation module with pluggable translator backends.

Provides an abstraction layer for different translation services
and implementations for mock and real translators.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

from glossary import GlossaryTerm
from models import TextChunk, TranslatedChunk
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """
    Abstract base class for translation implementations.

    Defines the interface that all translator implementations must follow.
    """

    @abstractmethod
    def translate(self, chunk: TextChunk) -> TranslatedChunk:
        """
        Translate a text chunk.

        Args:
            chunk: The TextChunk to translate

        Returns:
            A TranslatedChunk with the translated content
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this translator implementation.

        Returns:
            A string name (e.g., 'MockTranslator', 'OpenAITranslator')
        """
        raise NotImplementedError

    def configure_glossary(self, glossary_terms: Sequence[GlossaryTerm]) -> None:
        """
        Optional hook for glossary-aware translators.

        Default implementation is a no-op to keep compatibility across backends.
        """
        _ = glossary_terms


class MockTranslator(BaseTranslator):
    """
    Mock translator for testing and development.

    Returns a fake translated version by prefixing text with a marker.
    Allows full pipeline testing without API calls or API keys.
    """

    def __init__(self, target_language: str = "es"):
        """
        Initialize the mock translator.

        Args:
            target_language: Target language code (default: 'es' for Spanish)
        """
        self.target_language = target_language

    def translate(self, chunk: TextChunk) -> TranslatedChunk:
        """
        Return a mock translation of the chunk.

        For demonstration, prefixes the original text with a translation marker
        and translates a few known keywords.

        Args:
            chunk: The TextChunk to translate

        Returns:
            A TranslatedChunk with mock translated content
        """
        logger.info("MockTranslator: translating chunk %s", chunk.index)

        translated_text = self._mock_translate(chunk.original_text)

        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=translated_text,
            word_count=chunk.word_count,
            translator_used="MockTranslator",
            metadata={},
        )

    def _mock_translate(self, text: str) -> str:
        """
        Perform mock translation with simple keyword replacements.

        Args:
            text: Text to mock-translate

        Returns:
            Mock translated text
        """
        mock_dict = {
            "the": "el/la",
            "is": "es",
            "and": "y",
            "a": "un/una",
            "to": "a",
            "of": "de",
            "in": "en",
            "that": "ese/esa",
            "it": "lo/la",
            "for": "para",
        }

        translated = text
        for eng, esp in mock_dict.items():
            import re

            pattern = r"\b" + eng + r"\b"
            translated = re.sub(pattern, esp, translated, flags=re.IGNORECASE)

        return f"[ES MOCK TRANSLATION]\n{translated}"

    def get_name(self) -> str:
        """Return the name of this translator."""
        return "MockTranslator"


class OpenAITranslator(BaseTranslator):
    """
    OpenAI-based translator implementation.

    Uses the OpenAI Responses API to translate each chunk while preserving
    paragraph structure and document order.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",
        source_language: str = "English",
        target_language: str = "Spanish",
        glossary_terms: Optional[Sequence[GlossaryTerm]] = None,
    ):
        """
        Initialize the OpenAI translator.

        Args:
            api_key: OpenAI API key (optional if OPENAI_API_KEY env var exists)
            model: Model to use (default: gpt-5.2)
            source_language: Source language name
            target_language: Target language name
            glossary_terms: Optional glossary terms for prompt injection
        """
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
        )

    def translate(self, chunk: TextChunk) -> TranslatedChunk:
        """
        Translate a chunk using OpenAI API.

        Args:
            chunk: The TextChunk to translate

        Returns:
            A TranslatedChunk with translated content
        """
        logger.info("OpenAITranslator (%s): translating chunk %s", self.model, chunk.index)

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.prompt_builder.build_system_prompt(),
                input=self.prompt_builder.build_translation_prompt(chunk),
            )
            translated_text = self._extract_output_text(response)
            usage = self._extract_usage_metadata(response)

            if not translated_text:
                raise ValueError("OpenAI API returned an empty translation response")

            return TranslatedChunk(
                index=chunk.index,
                original_text=chunk.original_text,
                translated_text=translated_text,
                word_count=chunk.word_count,
                translator_used=f"OpenAITranslator({self.model})",
                estimated_tokens=usage.get("total_tokens"),
                metadata={"usage": usage} if usage else {},
            )

        except Exception as exc:
            logger.error("OpenAI translation failed for chunk %s: %s", chunk.index, exc)
            raise RuntimeError(
                f"OpenAI translation failed for chunk {chunk.index}: {exc}"
            ) from exc

    def configure_glossary(self, glossary_terms: Sequence[GlossaryTerm]) -> None:
        """Inject glossary terms into prompt generation at runtime."""
        self.prompt_builder.set_glossary_terms(glossary_terms)

    @staticmethod
    def _extract_output_text(response) -> str:
        """Extract text from a Responses API object."""
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
        """Extract token usage metadata from a Responses API object, if available."""
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
        """Convert short language codes to readable names for prompts."""
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
        """Return the name of this translator."""
        return f"OpenAITranslator({self.model})"


class TranslatorFactory:
    """
    Factory for creating translator instances.

    Supports pluggable translator types and future extensibility.
    """

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
    ) -> BaseTranslator:
        """
        Create a translator instance.

        Args:
            translator_type: Type of translator ('mock' or 'openai')
            source_language: Source language name
            target_language: Target language name
            api_key: Optional API key for translation service
            model: OpenAI model name when using OpenAI translator
            glossary_terms: Optional glossary terms for prompt-aware translators

        Returns:
            A BaseTranslator instance

        Raises:
            ValueError: If translator type is not supported
        """
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
            )

        raise ValueError(f"Cannot create translator: {translator_type}")

    @classmethod
    def register(cls, translator_type: str, translator_class: type) -> None:
        """
        Register a new translator type.

        Allows future extensibility for additional translator backends.

        Args:
            translator_type: Key for the translator type
            translator_class: Class implementing BaseTranslator
        """
        cls._translators[translator_type] = translator_class
        logger.info("Registered translator type: %s", translator_type)
