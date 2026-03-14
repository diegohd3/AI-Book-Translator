"""
Main entry point for the translation application.

Handles CLI argument parsing and pipeline execution.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

from config import DEFAULT_CACHE_DB_PATH, TranslationConfig
from pipeline import TranslationPipeline


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Translate TXT files from English to Spanish using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input book.txt --output book_es.txt
  python main.py --input book.txt --output book_es.txt --chunk-size 1500
  python main.py --input book.txt --output book_es.txt --translator openai --api-key sk-...
  python main.py --input book.txt --output book_es.txt --translator openai --glossary data/glossaries/example_fantasy_es.json --report data/output/report.json
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input TXT file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path for output TXT file",
    )

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
        "--report",
        type=str,
        default=None,
        help="Path to output JSON report for the run (optional)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Create configuration
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
            cache_db_path=Path(args.cache_db),
            disable_cache=args.disable_cache,
            disable_translation_memory=args.disable_tm,
            report_path=Path(args.report) if args.report else None,
        )

        # Run pipeline
        pipeline = TranslationPipeline(config)
        result = pipeline.run()

        # Print summary
        print("\n" + "=" * 60)
        print("TRANSLATION COMPLETE")
        print("=" * 60)
        print(f"Input file: {config.input_path}")
        print(f"Output file: {config.output_path}")
        print(f"Chunks processed: {result.total_chunks}")
        print(f"Translator used: {result.translator_used}")
        print(f"Source language: {result.source_language}")
        print(f"Target language: {result.target_language}")
        if config.report_path:
            print(f"Report file: {config.report_path}")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
