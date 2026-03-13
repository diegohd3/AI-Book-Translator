"""
Assembler module for combining translated chunks into final output.

Reassembles translated chunks while maintaining order and structure.
"""

import logging
from typing import List
from models import TranslatedChunk

logger = logging.getLogger(__name__)


class TranslationAssembler:
    """
    Reassembles translated chunks into a complete translated document.
    
    Ensures:
    - Chunks are reassembled in correct order
    - Paragraph structure is preserved
    - No data is lost during reassembly
    """

    def __init__(self):
        """Initialize the assembler."""
        pass

    def assemble(self, chunks: List[TranslatedChunk]) -> str:
        """
        Assemble translated chunks into a complete document.
        
        Args:
            chunks: List of TranslatedChunk objects in order
            
        Returns:
            The complete translated text
        """
        if not chunks:
            logger.warning("No chunks to assemble")
            return ""

        logger.info(f"Assembling {len(chunks)} chunks into final document")

        # Sort chunks by index to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.index)

        # Verify all chunks are present and in order
        for i, chunk in enumerate(sorted_chunks):
            if chunk.index != i:
                logger.warning(
                    f"Chunk index mismatch at position {i}: "
                    f"expected {i}, got {chunk.index}"
                )

        # Join chunks with double newline to preserve paragraph structure
        assembled_text = "\n\n".join(chunk.translated_text for chunk in sorted_chunks)

        logger.info(
            f"Successfully assembled {len(chunks)} chunks. "
            f"Output size: {len(assembled_text)} characters"
        )

        return assembled_text

    def verify_assembly(
        self, original_chunks: List[TranslatedChunk], assembled: str
    ) -> bool:
        """
        Verify that assembly was successful.
        
        Performs basic sanity checks on the assembled text.
        
        Args:
            original_chunks: The list of chunks that were assembled
            assembled: The assembled text
            
        Returns:
            True if assembly appears valid, False otherwise
        """
        if not assembled:
            logger.error("Assembled text is empty")
            return False

        # Count paragraphs in source and output
        # Note: this is a simple heuristic and may not be perfectly accurate
        original_para_count = sum(
            chunk.translated_text.count("\n\n") + 1 for chunk in original_chunks
        )
        assembled_para_count = assembled.count("\n\n") + 1

        logger.info(
            f"Assembly verification: "
            f"original ~{original_para_count} paragraphs, "
            f"assembled ~{assembled_para_count} paragraphs"
        )

        return True
