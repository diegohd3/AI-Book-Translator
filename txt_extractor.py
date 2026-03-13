"""
Text extraction module for reading TXT files from disk.

Handles file I/O with proper encoding and error handling.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TxtExtractor:
    """
    Extracts raw text from TXT files.
    
    Supports UTF-8 encoding with optional fallback strategies.
    """

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the extractor.
        
        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding

    def extract(self, file_path: Path) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            The complete text content of the file
            
        Raises:
            FileNotFoundError: If the file does not exist
            UnicodeDecodeError: If the file cannot be decoded with the specified encoding
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Input file does not exist: {file_path}")

        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info(f"Extracting text from: {file_path}")

        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                text = f.read()
            logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            return text

        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode file with {self.encoding}: {e}")
            raise UnicodeDecodeError(
                self.encoding,
                b"",
                0,
                1,
                f"Could not decode {file_path} with encoding {self.encoding}",
            ) from e
