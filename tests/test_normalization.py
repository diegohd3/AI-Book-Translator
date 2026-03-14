import unittest

from normalization import build_deterministic_key, normalize_text_for_lookup


class NormalizationTests(unittest.TestCase):
    def test_normalize_text_for_lookup(self) -> None:
        raw = "  Hello,\r\n\r\nworld!\tThis   is a test.  "
        normalized = normalize_text_for_lookup(raw)
        self.assertEqual("Hello, world! This is a test.", normalized)

    def test_deterministic_key_is_stable(self) -> None:
        key_a = build_deterministic_key("text", "en", "es", "gpt-5.2")
        key_b = build_deterministic_key("text", "en", "es", "gpt-5.2")
        key_c = build_deterministic_key("text", "en", "fr", "gpt-5.2")

        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)


if __name__ == "__main__":
    unittest.main()
