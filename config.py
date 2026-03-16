"""
Centralized configuration for the translation application.

Defines default values and settings that can be overridden at runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from document_format import AUTO_FORMAT, SUPPORTED_FORMATS, normalize_format_name
from style_profile import validate_style_profile_file


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
        glossary_path: Optional path to glossary JSON file
        style_profile_path: Optional path to style profile JSON file
        enable_refinement: Enable stage-2 literary refinement pass
        context_window: Number of recent chunk-context entries to inject into prompts
        cache_db_path: SQLite path used for cache and translation memory
        disable_cache: Disable cache lookups/storage when True
        disable_translation_memory: Disable translation memory lookups/storage when True
        report_output_path: Output path for JSON report sidecar
        input_format: Optional explicit input format ('txt', 'epub', or 'auto')
    """

    input_path: Path
    output_path: Path
    source_language: str = "en"
    target_language: str = "es"
    chunk_size: int = 1000
    translator_type: str = "mock"
    api_key: Optional[str] = None
    model: str = "gpt-5.2"
    glossary_path: Optional[Path] = None
    style_profile_path: Optional[Path] = None
    enable_refinement: bool = False
    context_window: int = 3
    cache_db_path: Path = Path("data/cache/translation_store.sqlite3")
    disable_cache: bool = False
    disable_translation_memory: bool = False
    report_output_path: Optional[Path] = None
    input_format: Optional[str] = None

    def __post_init__(self):
        """Validate configuration on creation."""
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.glossary_path, str):
            self.glossary_path = Path(self.glossary_path)
        if isinstance(self.style_profile_path, str):
            self.style_profile_path = Path(self.style_profile_path)
        if isinstance(self.cache_db_path, str):
            self.cache_db_path = Path(self.cache_db_path)
        if isinstance(self.report_output_path, str):
            self.report_output_path = Path(self.report_output_path)
        if isinstance(self.input_format, str):
            self.input_format = self.input_format.strip()

        if self.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 words")
        if self.chunk_size > 5000:
            raise ValueError("Chunk size should not exceed 5000 words")
        if self.translator_type not in ("mock", "openai"):
            raise ValueError("Translator type must be 'mock' or 'openai'")
        if not self.model or not self.model.strip():
            raise ValueError("Model name cannot be empty")
        if self.cache_db_path is None:
            raise ValueError("cache_db_path cannot be None")
        if self.context_window < 1:
            raise ValueError("context_window must be >= 1")
        normalized_format = normalize_format_name(self.input_format)
        if normalized_format and normalized_format != AUTO_FORMAT:
            if normalized_format not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"input_format must be one of: {', '.join(sorted(SUPPORTED_FORMATS))}, auto"
                )
        self.input_format = normalized_format

        if self.style_profile_path is not None:
            validate_style_profile_file(self.style_profile_path)

        if self.report_output_path is None:
            self.report_output_path = self.output_path.with_suffix(".report.json")

    def ensure_output_dir_exists(self) -> None:
        """Create output directory if it does not exist."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_report_output_dir_exists(self) -> None:
        """Create report directory if it does not exist."""
        if self.report_output_path is None:
            return
        self.report_output_path.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_SOURCE_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "es"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_TRANSLATOR = "mock"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_CACHE_DB_PATH = Path("data/cache/translation_store.sqlite3")
