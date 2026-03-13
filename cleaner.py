"""
Text cleaning and normalization module.

Performs basic text cleaning while preserving paragraph structure.
"""

import logging
import re

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes text while preserving paragraph boundaries.
    
    Handles:
    - Multiple consecutive blank lines
    - Trailing whitespace
    - Inconsistent line endings
    """

    def __init__(self, preserve_paragraphs: bool = True):
        """
        Initialize the cleaner.
        
        Args:
            preserve_paragraphs: Whether to preserve paragraph boundaries (default: True)
        """
        self.preserve_paragraphs = preserve_paragraphs

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with normalized spacing and encoding
        """
        if not text:
            logger.warning("Received empty text for cleaning")
            return ""

        logger.info("Cleaning text...")

        # Normalize line endings to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]

        # Collapse multiple consecutive blank lines into a single blank line
        # while preserving the paragraph structure
        if self.preserve_paragraphs:
            cleaned_lines = []
            prev_blank = False
            for line in lines:
                if line.strip() == "":
                    if not prev_blank:
                        cleaned_lines.append("")
                    prev_blank = True
                else:
                    cleaned_lines.append(line)
                    prev_blank = False
            text = "\n".join(cleaned_lines)
        else:
            text = "\n".join(lines)

        # Remove leading/trailing whitespace from entire document
        text = text.strip()

        logger.info(f"Text cleaned. Length: {len(text)} characters")
        return text
