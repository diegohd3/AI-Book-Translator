"""
SQLite-backed translation memory (TM) with exact-match retrieval.

Designed to support future fuzzy matching while keeping current behavior simple.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from normalization import build_deterministic_key, normalize_text_for_lookup
from sqlite_utils import ensure_sqlite_parent_dir, open_sqlite_row_connection

logger = logging.getLogger(__name__)


class TranslationMemoryError(RuntimeError):
    """Raised when translation memory operations fail."""


@dataclass
class TranslationMemoryMatch:
    """TM retrieval result."""

    translated_text: str
    metadata: Dict[str, object] = field(default_factory=dict)


class TranslationMemory:
    """SQLite-backed translation memory with exact normalized lookups."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        self._prepare_storage_path()

    def _prepare_storage_path(self) -> None:
        self.db_path = ensure_sqlite_parent_dir(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return open_sqlite_row_connection(self.db_path)

    def initialize_schema(self) -> None:
        """Create TM schema if it does not exist."""
        try:
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS translation_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_key TEXT NOT NULL UNIQUE,
                        normalized_source TEXT NOT NULL,
                        source_language TEXT NOT NULL,
                        target_language TEXT NOT NULL,
                        glossary_hash TEXT NOT NULL,
                        translated_text TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_used_at TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tm_lookup
                    ON translation_memory (
                        source_language, target_language, glossary_hash
                    );
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise TranslationMemoryError(
                f"Failed to initialize translation memory schema: {exc}"
            ) from exc

    def get_exact_match(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        glossary_hash: str,
    ) -> Optional[TranslationMemoryMatch]:
        """Return an exact TM match for normalized source text, if available."""
        normalized_source = normalize_text_for_lookup(source_text)
        memory_key = self._build_memory_key(
            normalized_source=normalized_source,
            source_language=source_language,
            target_language=target_language,
            glossary_hash=glossary_hash,
        )

        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    SELECT translated_text, metadata_json
                    FROM translation_memory
                    WHERE memory_key = ?
                    """,
                    (memory_key,),
                ).fetchone()

                if row is None:
                    return None

                conn.execute(
                    """
                    UPDATE translation_memory
                    SET last_used_at = CURRENT_TIMESTAMP
                    WHERE memory_key = ?
                    """,
                    (memory_key,),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise TranslationMemoryError(f"Failed to read TM entry: {exc}") from exc

        metadata = _parse_metadata_json(row["metadata_json"])
        return TranslationMemoryMatch(
            translated_text=str(row["translated_text"]),
            metadata=metadata,
        )

    def store_entry(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        glossary_hash: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Store or replace a TM entry for exact lookup reuse."""
        normalized_source = normalize_text_for_lookup(source_text)
        memory_key = self._build_memory_key(
            normalized_source=normalized_source,
            source_language=source_language,
            target_language=target_language,
            glossary_hash=glossary_hash,
        )
        metadata_json = (
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            if metadata is not None
            else None
        )

        try:
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO translation_memory (
                        memory_key,
                        normalized_source,
                        source_language,
                        target_language,
                        glossary_hash,
                        translated_text,
                        metadata_json,
                        last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(memory_key) DO UPDATE SET
                        translated_text = excluded.translated_text,
                        metadata_json = excluded.metadata_json,
                        last_used_at = CURRENT_TIMESTAMP
                    """,
                    (
                        memory_key,
                        normalized_source,
                        source_language,
                        target_language,
                        glossary_hash,
                        translated_text,
                        metadata_json,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise TranslationMemoryError(f"Failed to write TM entry: {exc}") from exc

    @staticmethod
    def _build_memory_key(
        *,
        normalized_source: str,
        source_language: str,
        target_language: str,
        glossary_hash: str,
    ) -> str:
        return build_deterministic_key(
            normalized_source,
            source_language.strip().lower(),
            target_language.strip().lower(),
            glossary_hash.strip(),
        )


def _parse_metadata_json(raw_value: Optional[str]) -> Dict[str, object]:
    """Parse metadata JSON defensively and return a dictionary."""
    if raw_value is None or not raw_value.strip():
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        logger.warning("TM metadata_json is invalid; ignoring stored metadata")
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}
