import json
import tempfile
import unittest
from pathlib import Path

from glossary import (
    GlossaryValidationError,
    find_relevant_terms,
    find_used_glossary_terms,
    load_glossary,
)


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

    def test_term_policy_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "glossary.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "terms": [
                            {"source": "Clocktower", "target": "Torre del reloj"},
                        ]
                    },
                    handle,
                )

            glossary = load_glossary(path)
            self.assertEqual("preferred", glossary.terms[0].policy)
            self.assertFalse(glossary.terms[0].case_sensitive)

    def test_relevant_and_used_matching(self) -> None:
        root = Path(__file__).resolve().parents[1]
        glossary = load_glossary(
            root / "data" / "glossaries" / "example_literary_es.json"
        )

        source = "Princess Donut moved near the Mana Core."
        relevant = find_relevant_terms(source, glossary.terms)

        self.assertEqual(2, len(relevant))
        relevant_sources = {item.term.source for item in relevant}
        self.assertIn("Princess Donut", relevant_sources)
        self.assertIn("Mana Core", relevant_sources)

        translated = "Princesa Donut se movió cerca del Núcleo de maná."
        used = find_used_glossary_terms(translated, relevant)
        self.assertIn("Princesa Donut", used)
        self.assertIn("Núcleo de maná", used)


if __name__ == "__main__":
    unittest.main()
