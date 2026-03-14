import unittest

from glossary import GlossaryMatch, GlossaryTerm
from models import TextChunk
from prompt_builder import PromptBuilder
from style_profile import default_style_profile


class PromptBuilderTests(unittest.TestCase):
    def test_builds_dialogue_prompt_with_glossary(self) -> None:
        term = GlossaryTerm(
            source="Princess Donut",
            target="Princesa Donut",
            policy="forced",
        )
        match = GlossaryMatch(term=term, count=1, positions=[(0, 15)])
        builder = PromptBuilder(
            source_language="English",
            target_language="Spanish",
            glossary_terms=[term],
            style_profile=default_style_profile("en", "es"),
        )

        chunk = TextChunk(index=0, original_text='"Open the door," Marta said.', word_count=5)
        prompt = builder.build_translation_prompt(
            chunk,
            stage="draft",
            chunk_signal="dialogue",
            context_snapshot={"recurring_character_names": ["Marta"]},
            relevant_matches=[match],
        )

        self.assertIn("SOURCE TEXT:", prompt)
        self.assertIn("FORCE", prompt)
        self.assertIn("Marta", prompt)

    def test_refinement_prompt_requires_draft(self) -> None:
        builder = PromptBuilder(style_profile=default_style_profile("en", "es"))
        chunk = TextChunk(index=0, original_text="Rain fell.", word_count=2)

        with self.assertRaises(ValueError):
            builder.build_translation_prompt(chunk, stage="refinement")


if __name__ == "__main__":
    unittest.main()
