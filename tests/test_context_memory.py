import unittest

from context_memory import BookContextMemory
from glossary import GlossaryMatch, GlossaryTerm
from style_profile import default_style_profile


class ContextMemoryTests(unittest.TestCase):
    def test_snapshot_tracks_recurring_terms(self) -> None:
        profile = default_style_profile("en", "es")
        memory = BookContextMemory(profile)

        term = GlossaryTerm(source="Mana Core", target="Núcleo de maná")
        match = GlossaryMatch(term=term, count=1, positions=[(0, 9)])

        memory.update_from_chunk(
            chunk_index=0,
            source_text="Marta touched the Mana Core in Seattle.",
            translated_text="Marta tocó el Núcleo de maná en Seattle.",
            chunk_signal="narration",
            glossary_matches=[match],
            glossary_used=["Núcleo de maná"],
            warnings=[],
        )
        memory.update_from_chunk(
            chunk_index=1,
            source_text="Marta saw the Mana Core again in Seattle.",
            translated_text="Marta volvió a ver el Núcleo de maná en Seattle.",
            chunk_signal="narration",
            glossary_matches=[match],
            glossary_used=["Núcleo de maná"],
            warnings=["RISK: sample"],
        )

        snapshot = memory.prompt_snapshot(context_window=1)
        recurring = memory.recurring_terms_summary()

        self.assertIn("Marta", recurring["characters"])
        self.assertIn("Seattle", recurring["places"])
        self.assertIn("RISK: sample", snapshot["translation_notes"])


if __name__ == "__main__":
    unittest.main()
