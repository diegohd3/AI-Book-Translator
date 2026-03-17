"""
Shared SQLite helpers used by cache-like persistence modules.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union


def ensure_sqlite_parent_dir(db_path: Union[str, Path]) -> str:
    """Ensure on-disk SQLite parent directory exists and return normalized path."""
    normalized_path = str(db_path)
    if normalized_path == ":memory:":
        return normalized_path

    path = Path(normalized_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def open_sqlite_row_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection configured with `sqlite3.Row` row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
