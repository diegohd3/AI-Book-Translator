"""
Main entry point for the translation application.

Handles CLI argument parsing and pipeline execution.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from config import DEFAULT_CACHE_DB_PATH, DEFAULT_CONTEXT_WINDOW, TranslationConfig
from pipeline import TranslationPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate TXT files with literary-aware translation intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input book.txt --output book_es.txt
  python main.py --input book.txt --output book_es.txt --translator openai --api-key sk-...
  python main.py --input book.txt --output book_es.txt --style-profile data/style_profiles/example_literary_es.json --glossary data/glossaries/example_literary_es.json --enable-refinement
  python main.py --input book.txt --output book_es.txt --report-output data/output/book_es.report.json
        """,
    )

    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input TXT file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for output TXT file")

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target word count per chunk (default: 1000)",
    )
    parser.add_argument(
        "--translator",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help="Translator backend to use (default: mock)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for translation service (if required)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "OpenAI model to use when --translator openai "
            "(default: OPENAI_MODEL or gpt-5.2)"
        ),
    )

    parser.add_argument(
        "--source-language",
        type=str,
        default="en",
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="es",
        help="Target language code (default: es)",
    )

    parser.add_argument(
        "--glossary",
        type=str,
        default=None,
        help="Path to glossary JSON file (optional)",
    )
    parser.add_argument(
        "--style-profile",
        type=str,
        default=None,
        help="Path to style profile JSON file (optional)",
    )
    parser.add_argument(
        "--enable-refinement",
        action="store_true",
        help="Enable stage-2 literary refinement pass",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"Chunk memory window for prompt context (default: {DEFAULT_CONTEXT_WINDOW})",
    )

    parser.add_argument(
        "--cache-db",
        type=str,
        default=str(DEFAULT_CACHE_DB_PATH),
        help=f"Path to SQLite DB used for cache/TM (default: {DEFAULT_CACHE_DB_PATH})",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable persistent translation cache",
    )
    parser.add_argument(
        "--disable-tm",
        action="store_true",
        help="Disable translation memory lookups/storage",
    )

    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Path to output JSON report (default: <output>.report.json)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Deprecated alias for --report-output",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        if args.report and args.report_output:
            logger.warning(
                "Both --report and --report-output were provided. "
                "Using --report-output."
            )
        report_output_value = args.report_output or args.report

        config = TranslationConfig(
            input_path=Path(args.input),
            output_path=Path(args.output),
            source_language=args.source_language,
            target_language=args.target_language,
            chunk_size=args.chunk_size,
            translator_type=args.translator,
            api_key=args.api_key,
            model=args.model or os.getenv("OPENAI_MODEL", "gpt-5.2"),
            glossary_path=Path(args.glossary) if args.glossary else None,
            style_profile_path=Path(args.style_profile) if args.style_profile else None,
            enable_refinement=args.enable_refinement,
            context_window=args.context_window,
            cache_db_path=Path(args.cache_db),
            disable_cache=args.disable_cache,
            disable_translation_memory=args.disable_tm,
            report_output_path=Path(report_output_value) if report_output_value else None,
        )

        pipeline = TranslationPipeline(config)
        result = pipeline.run()

        print("\n" + "=" * 60)
        print("TRANSLATION COMPLETE")
        print("=" * 60)
        print(f"Input file: {config.input_path}")
        print(f"Output file: {config.output_path}")
        print(f"Report file: {config.report_output_path}")
        print(f"Chunks processed: {result.total_chunks}")
        print(f"Translator used: {result.translator_used}")
        print(f"Source language: {result.source_language}")
        print(f"Target language: {result.target_language}")
        print(f"Refinement enabled: {config.enable_refinement}")
        print("=" * 60)

        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
