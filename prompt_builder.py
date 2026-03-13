"""
Prompt builder module for constructing translation requests.

Creates prompts and request payloads for translation services.
"""

import logging
from models import TextChunk

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds translation prompts and request payloads.
    
    Handles formatting of text chunks into structured prompts
    suitable for translation API calls.
    """

    def __init__(
        self, source_language: str = "English", target_language: str = "Spanish"
    ):
        """
        Initialize the prompt builder.
        
        Args:
            source_language: Name of the source language (default: English)
            target_language: Name of the target language (default: Spanish)
        """
        self.source_language = source_language
        self.target_language = target_language

    def build_translation_prompt(self, chunk: TextChunk) -> str:
        """
        Build a translation prompt for a text chunk.
        
        Args:
            chunk: The TextChunk to translate
            
        Returns:
            A formatted prompt string
        """
        prompt = (
            f"Translate the following text from {self.source_language} "
            f"to {self.target_language}. "
            f"Preserve the original formatting and paragraph structure.\n\n"
            f"TEXT TO TRANSLATE:\n"
            f"{chunk.original_text}\n\n"
            f"TRANSLATION:"
        )
        return prompt

    def build_system_prompt(self) -> str:
        """
        Build a system-level prompt for translation context.
        
        Returns:
            A system prompt for translation tasks
        """
        return (
            f"You are a professional translator specializing in high-quality "
            f"literary translation from {self.source_language} to {self.target_language}. "
            f"Maintain the tone, style, and meaning of the original text. "
            f"Preserve paragraph structure and formatting."
        )

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
