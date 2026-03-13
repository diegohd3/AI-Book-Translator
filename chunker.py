"""
Text chunking module for splitting text into manageable segments.

Preserves paragraph boundaries when creating chunks.
"""

import logging
from typing import List
from models import TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into chunks based on paragraph boundaries and word count.
    
    Ensures:
    - Paragraphs are never split in the middle
    - Target chunk size is respected when possible
    - Single large paragraphs are allowed to exceed target size
    """

    def __init__(self, target_chunk_size: int = 1000):
        """
        Initialize the chunker.
        
        Args:
            target_chunk_size: Target word count for each chunk (default: 1000)
        """
        if target_chunk_size < 100:
            raise ValueError("Target chunk size must be at least 100 words")
        self.target_chunk_size = target_chunk_size

    def chunk(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks based on paragraph boundaries.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of TextChunk objects in order
        """
        if not text or not text.strip():
            logger.warning("Received empty text for chunking")
            return []

        logger.info(f"Chunking text with target size: {self.target_chunk_size} words")

        # Split by blank lines to get paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if p]  # Remove empty paragraphs

        if not paragraphs:
            logger.warning("No paragraphs found after splitting")
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0

        for para in paragraphs:
            para_word_count = self._count_words(para)

            # If we're starting fresh and this paragraph itself is larger than target,
            # allow it as its own chunk
            if not current_chunk:
                current_chunk.append(para)
                current_word_count = para_word_count

                # If paragraph alone exceeds target, wrap it now
                if para_word_count >= self.target_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk, current_word_count, len(chunks)
                        )
                    )
                    current_chunk = []
                    current_word_count = 0

            # If adding this paragraph would exceed target, save current chunk
            elif current_word_count + para_word_count > self.target_chunk_size:
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            current_chunk, current_word_count, len(chunks)
                        )
                    )
                current_chunk = [para]
                current_word_count = para_word_count

                # If this paragraph alone exceeds target, wrap it now
                if para_word_count >= self.target_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk, current_word_count, len(chunks)
                        )
                    )
                    current_chunk = []
                    current_word_count = 0

            # Otherwise, add paragraph to current chunk
            else:
                current_chunk.append(para)
                current_word_count += para_word_count

        # Append any remaining chunk
        if current_chunk:
            chunks.append(
                self._create_chunk(current_chunk, current_word_count, len(chunks))
            )

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    def _create_chunk(
        self, paragraphs: List[str], word_count: int, index: int
    ) -> TextChunk:
        """
        Create a TextChunk from a list of paragraphs.
        
        Args:
            paragraphs: List of paragraph strings
            word_count: Total word count in the chunk
            index: Index of the chunk
            
        Returns:
            A TextChunk object
        """
        text = "\n\n".join(paragraphs)
        return TextChunk(index=index, original_text=text, word_count=word_count)

    @staticmethod
    def _count_words(text: str) -> int:
        """
        Count words in text using simple whitespace splitting.
        
        Args:
            text: Text to count
            
        Returns:
            Word count
        """
        return len(text.split())
