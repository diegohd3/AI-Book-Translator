"""
SQLite-backed persistent translation cache.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

from normalization import build_deterministic_key, normalize_text_for_lookup

logger = logging.getLogger(__name__)


class CacheError(RuntimeError):
    """Raised when cache operations fail."""


class TranslationCache:
    """
    Persistent translation cache keyed by normalized source and config dimensions.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        self._prepare_storage_path()

    def _prepare_storage_path(self) -> None:
        """Create parent directory for SQLite file when needed."""
        if self.db_path == ":memory:":
            return
        path = Path(self.db_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row factory configured."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        """Create cache schema if it does not exist."""
        statement = """
        CREATE TABLE IF NOT EXISTS translation_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT NOT NULL UNIQUE,
            normalized_source TEXT NOT NULL,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            model TEXT NOT NULL,
            glossary_hash TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            with self._connect() as conn:
                conn.execute(statement)
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cache_lookup
                    ON translation_cache (
                        source_language, target_language, model, glossary_hash
                    );
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Failed to initialize cache schema: {exc}") from exc

    def get_cached_translation(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        model: str,
        glossary_hash: str,
    ) -> Optional[str]:
        """Return cached translation for an exact normalized key, if present."""
        normalized_source = normalize_text_for_lookup(source_text)
        cache_key = self._build_cache_key(
            normalized_source=normalized_source,
            source_language=source_language,
            target_language=target_language,
            model=model,
            glossary_hash=glossary_hash,
        )

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT translated_text FROM translation_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise CacheError(f"Failed to read cache entry: {exc}") from exc

        if row is None:
            return None
        return str(row["translated_text"])

    def store_cached_translation(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        model: str,
        glossary_hash: str,
    ) -> None:
        """Store or replace cache entry for a deterministic key."""
        normalized_source = normalize_text_for_lookup(source_text)
        cache_key = self._build_cache_key(
            normalized_source=normalized_source,
            source_language=source_language,
            target_language=target_language,
            model=model,
            glossary_hash=glossary_hash,
        )

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO translation_cache (
                        cache_key,
                        normalized_source,
                        source_language,
                        target_language,
                        model,
                        glossary_hash,
                        translated_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        normalized_source,
                        source_language,
                        target_language,
                        model,
                        glossary_hash,
                        translated_text,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Failed to write cache entry: {exc}") from exc

    @staticmethod
    def _build_cache_key(
        *,
        normalized_source: str,
        source_language: str,
        target_language: str,
        model: str,
        glossary_hash: str,
    ) -> str:
        return build_deterministic_key(
            normalized_source,
            source_language.strip().lower(),
            target_language.strip().lower(),
            model.strip(),
            glossary_hash.strip(),
        )
