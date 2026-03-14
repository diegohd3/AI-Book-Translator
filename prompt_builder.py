"""
Prompt builder module for constructing translation requests.

Creates prompts and request payloads for translation services.
"""

import logging
from typing import Optional, Sequence

from glossary import GlossaryTerm, iter_glossary_pairs
from models import TextChunk

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds translation prompts and request payloads.
    
    Handles formatting of text chunks into structured prompts
    suitable for translation API calls.
    """

    def __init__(
        self,
        source_language: str = "English",
        target_language: str = "Spanish",
        glossary_terms: Optional[Sequence[GlossaryTerm]] = None,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            source_language: Name of the source language (default: English)
            target_language: Name of the target language (default: Spanish)
            glossary_terms: Optional glossary terms to inject into prompts
        """
        self.source_language = source_language
        self.target_language = target_language
        self.glossary_terms = list(glossary_terms or [])

    def set_glossary_terms(self, glossary_terms: Optional[Sequence[GlossaryTerm]]) -> None:
        """Update glossary terms used for prompt construction."""
        self.glossary_terms = list(glossary_terms or [])

    def build_translation_prompt(self, chunk: TextChunk) -> str:
        """
        Build a translation prompt for a text chunk.
        
        Args:
            chunk: The TextChunk to translate
            
        Returns:
            A formatted prompt string
        """
        prompt_parts = [
            f"Translate the following text from {self.source_language} to {self.target_language}.",
            "Preserve narrative style, meaning, and paragraph structure.",
            "Return only the translation.",
        ]

        glossary_block = self._build_glossary_block()
        if glossary_block:
            prompt_parts.append(glossary_block)

        prompt_parts.extend(
            [
                "",
                "TEXT TO TRANSLATE:",
                chunk.original_text,
                "",
                "TRANSLATION:",
            ]
        )

        prompt = "\n".join(prompt_parts)
        return prompt

    def build_system_prompt(self) -> str:
        """
        Build a system-level prompt for translation context.
        
        Returns:
            A system prompt for translation tasks
        """
        system_prompt = (
            f"You are a professional translator specializing in high-quality "
            f"literary translation from {self.source_language} to {self.target_language}. "
            f"Maintain the tone, style, and meaning of the original text. "
            f"Preserve paragraph structure and formatting."
        )
        if self.glossary_terms:
            system_prompt += " Follow provided glossary mappings whenever source terms appear."
        return system_prompt

    def build_payload(
        self, chunk: TextChunk, api_format: str = "openai"
    ) -> dict:
        """
        Build an API request payload for a translation service.
        
        Args:
            chunk: The TextChunk to translate
            api_format: Format of the payload ('openai' or other, default: 'openai')
            
        Returns:
            A dictionary suitable for API requests
        """
        if api_format == "openai":
            return {
                "model": "gpt-5.2",
                "instructions": self.build_system_prompt(),
                "input": self.build_translation_prompt(chunk),
            }
        else:
            # Generic format for future API implementations
            return {
                "source_language": self.source_language,
                "target_language": self.target_language,
                "text": chunk.original_text,
                "chunk_index": chunk.index,
            }

    def _build_glossary_block(self) -> str:
        """Build deterministic glossary instructions for prompt injection."""
        if not self.glossary_terms:
            return ""

        lines = [
            "Glossary constraints (apply exactly when terms appear):",
        ]
        for source, target, notes in iter_glossary_pairs(self.glossary_terms):
            line = f'- "{source}" -> "{target}"'
            if notes:
                line = f"{line} ({notes})"
            lines.append(line)

        return "\n".join(lines)
