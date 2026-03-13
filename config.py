"""
Centralized configuration for the translation application.

Defines default values and settings that can be overridden at runtime.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationConfig:
    """
    Configuration for translation operations.
    
    Attributes:
        input_path: Path to the input TXT file
        output_path: Path for the output TXT file
        source_language: Language code of input (default: 'en')
        target_language: Language code of output (default: 'es')
        chunk_size: Target word count for each chunk (default: 1000)
        translator_type: Which translator to use ('mock' or 'openai', default: 'mock')
        api_key: API key for paid translation services (optional)
        model: OpenAI model name when using OpenAI translator (default: 'gpt-5.2')
    """
    input_path: Path
    output_path: Path
    source_language: str = "en"
    target_language: str = "es"
    chunk_size: int = 1000
    translator_type: str = "mock"
    api_key: Optional[str] = None
    model: str = "gpt-5.2"

    def __post_init__(self):
        """Validate configuration on creation."""
        # Ensure paths are Path objects
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Validate values
        if self.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 words")
        if self.chunk_size > 5000:
            raise ValueError("Chunk size should not exceed 5000 words")
        if self.translator_type not in ("mock", "openai"):
            raise ValueError("Translator type must be 'mock' or 'openai'")
        if not self.model or not self.model.strip():
            raise ValueError("Model name cannot be empty")

    def ensure_output_dir_exists(self) -> None:
        """Create output directory if it does not exist."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


# Default configuration values
DEFAULT_SOURCE_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "es"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_TRANSLATOR = "mock"
DEFAULT_MODEL = "gpt-5.2"
