import json
import tempfile
import unittest
from pathlib import Path

from glossary import GlossaryValidationError, load_glossary


class GlossaryTests(unittest.TestCase):
    def test_load_example_glossary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        glossary_path = root / "data" / "glossaries" / "example_fantasy_es.json"

        glossary = load_glossary(glossary_path)

        self.assertEqual(2, len(glossary.terms))
        self.assertNotEqual("no_glossary", glossary.glossary_hash())

    def test_invalid_glossary_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid_glossary.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"not_terms": []}, handle)

            with self.assertRaises(GlossaryValidationError):
                load_glossary(path)


if __name__ == "__main__":
    unittest.main()
