"""
Translation run reporting utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranslationRunReport:
    """Structured report payload for a translation execution."""

    input_file: str
    output_file: str
    translator_backend: str
    model: str
    source_language: str
    target_language: str
    total_chunks: int
    translated_chunks: int
    cache_hits: int
    cache_misses: int
    translation_memory_hits: int
    estimated_token_usage: Optional[int]
    started_at: str
    finished_at: str
    elapsed_seconds: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to a serializable dictionary."""
        return asdict(self)


class ReportWriter:
    """Writes JSON reports to disk in UTF-8."""

    @staticmethod
    def write(report: TranslationRunReport, output_path: Path) -> None:
        """Persist report to JSON file, creating parent folders automatically."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = report.to_dict()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        logger.info("Translation report written to: %s", output_path)
