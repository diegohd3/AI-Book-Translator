"""
Data models for the translation pipeline.

Defines dataclasses for document chunks, translation results, and related structures.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TextChunk:
    """
    Represents a chunk of text to be translated.
    
    Attributes:
        index: Position of this chunk in the sequence
        original_text: The raw text content of the chunk
        word_count: Number of words in the chunk
    """
    index: int
    original_text: str
    word_count: int

    def __post_init__(self):
        """Validate chunk on creation."""
        if self.index < 0:
            raise ValueError("Chunk index must be non-negative")
        if not self.original_text:
            raise ValueError("Chunk text cannot be empty")
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")


@dataclass
class TranslatedChunk:
    """
    Represents a translated chunk.
    
    Attributes:
        index: Position of this chunk in the sequence
        original_text: The original text before translation
        translated_text: The translated text
        word_count: Number of words in the original chunk
        translator_used: Name of the translator that produced this translation
    """
    index: int
    original_text: str
    translated_text: str
    word_count: int
    translator_used: str = "unknown"


@dataclass
class TranslationResult:
    """
    Contains the complete result of a translation operation.
    
    Attributes:
        source_language: Language code of the source text (e.g., 'en')
        target_language: Language code of the target text (e.g., 'es')
        original_text: The complete original text
        translated_text: The complete translated text
        chunks: All individual translated chunks in order
        total_chunks: Total number of chunks processed
        translator_used: Which translator was used for this round
    """
    source_language: str
    target_language: str
    original_text: str
    translated_text: str
    chunks: List[TranslatedChunk] = field(default_factory=list)
    total_chunks: int = 0
    translator_used: str = "unknown"

    def __post_init__(self):
        """Validate result on creation."""
        if not self.source_language or not self.target_language:
            raise ValueError("Source and target languages must be specified")
        if self.total_chunks > 0 and len(self.chunks) != self.total_chunks:
            raise ValueError("Number of chunks does not match total_chunks count")
