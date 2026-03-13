"""
Translation module with pluggable translator backends.

Provides an abstraction layer for different translation services
and implementations for mock and real translators.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional
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
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this translator implementation.
        
        Returns:
            A string name (e.g., 'MockTranslator', 'OpenAITranslator')
        """
        pass


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
        logger.info(f"MockTranslator: translating chunk {chunk.index}")

        # Simple mock translation: prefix with language code and perform
        # basic keyword substitution for demonstration
        translated_text = self._mock_translate(chunk.original_text)

        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=translated_text,
            word_count=chunk.word_count,
            translator_used="MockTranslator",
        )

    def _mock_translate(self, text: str) -> str:
        """
        Perform mock translation with simple keyword replacements.
        
        Args:
            text: Text to mock-translate
            
        Returns:
            Mock translated text
        """
        # Basic mock translation dictionary for common English words
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

        # Perform simple case-insensitive replacements
        translated = text
        for eng, esp in mock_dict.items():
            # Replace whole words only (simple approach)
            import re

            pattern = r"\b" + eng + r"\b"
            translated = re.sub(pattern, esp, translated, flags=re.IGNORECASE)

        # Add a prefix to indicate this is a mock translation
        translated = f"[ES MOCK TRANSLATION]\n{translated}"
        return translated

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
    ):
        """
        Initialize the OpenAI translator.
        
        Args:
            api_key: OpenAI API key (optional if OPENAI_API_KEY env var exists)
            model: Model to use (default: gpt-5.2)
            source_language: Source language name
            target_language: Target language name
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
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from e

        self.client = OpenAI(api_key=self.api_key)
        self.prompt_builder = PromptBuilder(
            source_language=self.source_language,
            target_language=self.target_language,
        )

    def translate(self, chunk: TextChunk) -> TranslatedChunk:
        """
        Translate a chunk using OpenAI API.
        
        Args:
            chunk: The TextChunk to translate
            
        Returns:
            A TranslatedChunk with translated content
        """
        logger.info(
            f"OpenAITranslator ({self.model}): translating chunk {chunk.index}"
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.prompt_builder.build_system_prompt(),
                input=self.prompt_builder.build_translation_prompt(chunk),
            )
            translated_text = self._extract_output_text(response)

            if not translated_text:
                raise ValueError("OpenAI API returned an empty translation response")

            return TranslatedChunk(
                index=chunk.index,
                original_text=chunk.original_text,
                translated_text=translated_text,
                word_count=chunk.word_count,
                translator_used=f"OpenAITranslator({self.model})",
            )

        except Exception as e:
            logger.error(f"OpenAI translation failed for chunk {chunk.index}: {e}")
            raise RuntimeError(
                f"OpenAI translation failed for chunk {chunk.index}: {e}"
            ) from e

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
    ) -> BaseTranslator:
        """
        Create a translator instance.
        
        Args:
            translator_type: Type of translator ('mock' or 'openai')
            source_language: Source language name
            target_language: Target language name
            api_key: Optional API key for translation service
            model: OpenAI model name when using OpenAI translator
            
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
        elif translator_type == "openai":
            return translator_class(
                api_key=api_key,
                model=model,
                source_language=source_language,
                target_language=target_language,
            )
        
        raise ValueError(f"Cannot create translator: {translator_type}")

    @classmethod
    def register(
        cls, translator_type: str, translator_class: type
    ) -> None:
        """
        Register a new translator type.
        
        Allows future extensibility for additional translator backends.
        
        Args:
            translator_type: Key for the translator type
            translator_class: Class implementing BaseTranslator
        """
        cls._translators[translator_type] = translator_class
        logger.info(f"Registered translator type: {translator_type}")
