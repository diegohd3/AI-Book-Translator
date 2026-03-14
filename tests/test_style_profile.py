import json
import tempfile
import unittest
from pathlib import Path

from style_profile import (
    StyleProfileValidationError,
    default_style_profile,
    load_style_profile,
)


class StyleProfileTests(unittest.TestCase):
    def test_default_profile_has_hash(self) -> None:
        profile = default_style_profile("en", "es")
        self.assertTrue(profile.profile_hash())

    def test_load_valid_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "style_profiles" / "example_literary_es.json"
        profile = load_style_profile(path)
        self.assertEqual("en", profile.source_language)
        self.assertGreater(len(profile.forbidden_patterns), 0)

    def test_missing_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_style.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"source_language": "en"}, handle)

            with self.assertRaises(StyleProfileValidationError):
                load_style_profile(path)


if __name__ == "__main__":
    unittest.main()
