import tempfile
import unittest
from pathlib import Path

from cache import TranslationCache


class TranslationCacheTests(unittest.TestCase):
    def test_store_and_retrieve_cached_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "cache.sqlite3"
            cache = TranslationCache(db_path)
            cache.initialize_schema()

            source_text = "Mana Core is stable."
            translated_text = "El Nucleo de mana es estable."

            cache.store_cached_translation(
                source_text=source_text,
                translated_text=translated_text,
                source_language="en",
                target_language="es",
                model="gpt-5.2",
                glossary_hash="hash-1",
            )

            cached = cache.get_cached_translation(
                source_text="  Mana   Core is  stable.\n",
                source_language="en",
                target_language="es",
                model="gpt-5.2",
                glossary_hash="hash-1",
            )

            self.assertEqual(translated_text, cached)

    def test_cache_miss_with_different_glossary_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "cache.sqlite3"
            cache = TranslationCache(db_path)
            cache.initialize_schema()

            cache.store_cached_translation(
                source_text="sword aura",
                translated_text="aura de espada",
                source_language="en",
                target_language="es",
                model="gpt-5.2",
                glossary_hash="hash-a",
            )

            cached = cache.get_cached_translation(
                source_text="sword aura",
                source_language="en",
                target_language="es",
                model="gpt-5.2",
                glossary_hash="hash-b",
            )
            self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
