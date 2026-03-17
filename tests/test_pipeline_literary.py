import json
import tempfile
import unittest
from pathlib import Path

from config import TranslationConfig
from pipeline import TranslationPipeline


class LiteraryPipelineTests(unittest.TestCase):
    def test_pipeline_generates_literary_report_with_mock_refinement_noop(self) -> None:
        root = Path(__file__).resolve().parents[1]
        input_text = root / "data" / "input" / "sample_literary_excerpt.txt"
        style_path = root / "data" / "style_profiles" / "example_literary_es.json"
        glossary_path = root / "data" / "glossaries" / "example_literary_es.json"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "book_es.txt"
            report_path = tmp_path / "book_es.report.json"
            db_path = tmp_path / "cache.sqlite3"

            config = TranslationConfig(
                input_path=input_text,
                output_path=output_path,
                source_language="en",
                target_language="es",
                translator_type="mock",
                glossary_path=glossary_path,
                style_profile_path=style_path,
                enable_refinement=True,
                context_window=2,
                cache_db_path=db_path,
                report_output_path=report_path,
            )

            pipeline = TranslationPipeline(config)
            result = pipeline.run()

            self.assertGreater(result.total_chunks, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)

            self.assertIn("style_profile_hash", report)
            self.assertIn("policy_hash", report)
            self.assertIn("chunks", report)
            self.assertGreaterEqual(len(report["chunks"]), 1)
            self.assertFalse(report["chunks"][0]["refinement_applied"])
            self.assertEqual([], report["consistency_risks"])
            self.assertEqual([], report["chunks"][0]["warnings"])

    def test_policy_hash_changes_cache_partition(self) -> None:
        root = Path(__file__).resolve().parents[1]
        input_text = root / "data" / "input" / "sample_literary_excerpt.txt"
        style_path = root / "data" / "style_profiles" / "example_literary_es.json"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "cache.sqlite3"

            config_a = TranslationConfig(
                input_path=input_text,
                output_path=tmp_path / "a.txt",
                translator_type="mock",
                style_profile_path=style_path,
                context_window=2,
                cache_db_path=db_path,
                report_output_path=tmp_path / "a.report.json",
            )
            TranslationPipeline(config_a).run()

            config_b = TranslationConfig(
                input_path=input_text,
                output_path=tmp_path / "b.txt",
                translator_type="mock",
                style_profile_path=style_path,
                context_window=5,
                cache_db_path=db_path,
                report_output_path=tmp_path / "b.report.json",
            )
            TranslationPipeline(config_b).run()

            with open(config_b.report_output_path, "r", encoding="utf-8") as handle:
                report_b = json.load(handle)

            self.assertEqual(0, report_b["cache_hits"])


if __name__ == "__main__":
    unittest.main()
