"""
Main orchestration pipeline for the translation workflow.

Coordinates all modules to perform end-to-end document translation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from assembler import TranslationAssembler
from cache import CacheError, TranslationCache
from chunker import TextChunker
from cleaner import TextCleaner
from config import TranslationConfig
from glossary import Glossary, GlossaryValidationError, load_glossary
from models import TextChunk, TranslatedChunk, TranslationResult
from reporting import ReportWriter, TranslationRunReport
from translation_memory import (
    TranslationMemory,
    TranslationMemoryError,
    TranslationMemoryMatch,
)
from translator import BaseTranslator, TranslatorFactory
from txt_extractor import TxtExtractor

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    End-to-end translation pipeline with quality/reuse layers.

    Workflow:
    1. Load glossary (optional)
    2. Initialize persistent cache and translation memory
    3. Extract, clean, and chunk input text
    4. For each chunk: cache lookup -> TM lookup -> translate on miss
    5. Persist new translations into cache/TM
    6. Assemble final output and write file
    7. Optionally write run report JSON
    """

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.extractor = TxtExtractor(encoding="utf-8")
        self.cleaner = TextCleaner(preserve_paragraphs=True)
        self.chunker = TextChunker(target_chunk_size=config.chunk_size)
        self.assembler = TranslationAssembler()
        self.translator = self._create_translator()

    def _create_translator(self) -> BaseTranslator:
        """Create a translator instance based on configuration."""
        return TranslatorFactory.create(
            translator_type=self.config.translator_type,
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            api_key=self.config.api_key,
            model=self.config.model,
        )

    def run(self) -> TranslationResult:
        """
        Execute the complete translation pipeline.

        Returns:
            A TranslationResult containing translated content and metadata.
        """
        started_at_dt = datetime.now(timezone.utc)
        started_at = started_at_dt.isoformat()
        errors: List[str] = []

        total_chunks = 0
        cache_hits = 0
        cache_misses = 0
        tm_hits = 0
        original_text = ""
        translated_chunks: List[TranslatedChunk] = []

        logger.info("=" * 60)
        logger.info("Starting translation pipeline")
        logger.info("Input: %s", self.config.input_path)
        logger.info("Output: %s", self.config.output_path)
        logger.info("Translator: %s", self.translator.get_name())
        logger.info("=" * 60)

        glossary_hash = "no_glossary"
        cache_layer: Optional[TranslationCache] = None
        tm_layer: Optional[TranslationMemory] = None

        try:
            glossary = self._load_glossary()
            glossary_hash = glossary.glossary_hash()
            self.translator.configure_glossary(glossary.terms)

            cache_layer = self._initialize_cache(errors)
            tm_layer = self._initialize_translation_memory(errors)

            logger.info("Step 1: Extracting text from file...")
            original_text = self.extractor.extract(self.config.input_path)

            logger.info("Step 2: Cleaning text...")
            cleaned_text = self.cleaner.clean(original_text)

            logger.info("Step 3: Chunking text...")
            chunks = self.chunker.chunk(cleaned_text)
            if not chunks:
                raise ValueError("No chunks were created from the text")
            total_chunks = len(chunks)
            logger.info("Created %s chunks", total_chunks)

            logger.info("Step 4-5: Reusing translations and translating misses...")
            for sequence_index, chunk in enumerate(chunks, start=1):
                logger.info(
                    "  Processing chunk %s/%s (%s words)...",
                    sequence_index,
                    total_chunks,
                    chunk.word_count,
                )

                translated = None

                if cache_layer is not None:
                    try:
                        cached_text = cache_layer.get_cached_translation(
                            source_text=chunk.original_text,
                            source_language=self.config.source_language,
                            target_language=self.config.target_language,
                            model=self._cache_model_key(),
                            glossary_hash=glossary_hash,
                        )
                        if cached_text is not None:
                            cache_hits += 1
                            translated = self._build_reused_chunk(
                                chunk=chunk,
                                translated_text=cached_text,
                                source_tag="cache",
                                metadata={"glossary_hash": glossary_hash},
                            )
                    except CacheError as exc:
                        message = f"Cache lookup failed for chunk {chunk.index}: {exc}"
                        logger.warning(message)
                        errors.append(message)

                if translated is None and cache_layer is not None:
                    cache_misses += 1

                if translated is None and tm_layer is not None:
                    try:
                        tm_match = tm_layer.get_exact_match(
                            source_text=chunk.original_text,
                            source_language=self.config.source_language,
                            target_language=self.config.target_language,
                            glossary_hash=glossary_hash,
                        )
                        if tm_match is not None:
                            tm_hits += 1
                            translated = self._build_tm_chunk(chunk, tm_match)
                            if cache_layer is not None:
                                self._safe_store_cache(
                                    cache_layer=cache_layer,
                                    chunk=chunk,
                                    translated_text=translated.translated_text,
                                    glossary_hash=glossary_hash,
                                    errors=errors,
                                )
                    except TranslationMemoryError as exc:
                        message = (
                            f"Translation memory lookup failed for chunk {chunk.index}: {exc}"
                        )
                        logger.warning(message)
                        errors.append(message)

                if translated is None:
                    translated = self.translator.translate(chunk)
                    translated.translation_source = "translator"

                    if cache_layer is not None:
                        self._safe_store_cache(
                            cache_layer=cache_layer,
                            chunk=chunk,
                            translated_text=translated.translated_text,
                            glossary_hash=glossary_hash,
                            errors=errors,
                        )
                    if tm_layer is not None:
                        self._safe_store_tm(
                            tm_layer=tm_layer,
                            chunk=chunk,
                            translated_chunk=translated,
                            glossary_hash=glossary_hash,
                            errors=errors,
                        )

                translated_chunks.append(translated)

            logger.info("Step 6: Assembling translated chunks...")
            final_translated_text = self.assembler.assemble(translated_chunks)

            logger.info("Step 7: Writing output file...")
            self.config.ensure_output_dir_exists()
            self._write_output(final_translated_text)

            logger.info("=" * 60)
            logger.info("Translation pipeline completed successfully")
            logger.info("=" * 60)

            return TranslationResult(
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                original_text=original_text,
                translated_text=final_translated_text,
                chunks=translated_chunks,
                total_chunks=total_chunks,
                translator_used=self.translator.get_name(),
            )
        except Exception as exc:
            message = f"Pipeline failed with error: {exc}"
            logger.error(message, exc_info=True)
            errors.append(message)
            raise
        finally:
            self._write_report(
                started_at=started_at,
                started_at_dt=started_at_dt,
                total_chunks=total_chunks,
                translated_chunks=translated_chunks,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                tm_hits=tm_hits,
                errors=errors,
            )

    def _load_glossary(self) -> Glossary:
        """Load optional glossary from configuration."""
        if self.config.glossary_path is None:
            logger.info("No glossary provided; continuing without glossary constraints")
            return Glossary(terms=[])

        logger.info("Loading glossary from: %s", self.config.glossary_path)
        try:
            glossary = load_glossary(self.config.glossary_path)
        except (FileNotFoundError, GlossaryValidationError) as exc:
            raise ValueError(f"Invalid glossary configuration: {exc}") from exc

        logger.info(
            "Glossary enabled (%s terms, hash=%s)",
            len(glossary.terms),
            glossary.glossary_hash(),
        )
        return glossary

    def _initialize_cache(self, errors: List[str]) -> Optional[TranslationCache]:
        """Create and initialize cache layer unless disabled."""
        if self.config.disable_cache:
            logger.info("Cache layer disabled by configuration")
            return None

        cache_layer = TranslationCache(self.config.cache_db_path)
        try:
            cache_layer.initialize_schema()
        except CacheError as exc:
            message = f"Cache initialization failed; continuing without cache: {exc}"
            logger.warning(message)
            errors.append(message)
            return None

        logger.info("Cache enabled with DB: %s", self.config.cache_db_path)
        return cache_layer

    def _initialize_translation_memory(
        self, errors: List[str]
    ) -> Optional[TranslationMemory]:
        """Create and initialize translation memory unless disabled."""
        if self.config.disable_translation_memory:
            logger.info("Translation memory layer disabled by configuration")
            return None

        tm_layer = TranslationMemory(self.config.cache_db_path)
        try:
            tm_layer.initialize_schema()
        except TranslationMemoryError as exc:
            message = (
                "Translation memory initialization failed; continuing without TM: "
                f"{exc}"
            )
            logger.warning(message)
            errors.append(message)
            return None

        logger.info("Translation memory enabled with DB: %s", self.config.cache_db_path)
        return tm_layer

    def _safe_store_cache(
        self,
        cache_layer: TranslationCache,
        chunk: TextChunk,
        translated_text: str,
        glossary_hash: str,
        errors: List[str],
    ) -> None:
        """Store cache entry and continue on failures."""
        try:
            cache_layer.store_cached_translation(
                source_text=chunk.original_text,
                translated_text=translated_text,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                model=self._cache_model_key(),
                glossary_hash=glossary_hash,
            )
        except CacheError as exc:
            message = f"Cache store failed for chunk {chunk.index}: {exc}"
            logger.warning(message)
            errors.append(message)

    def _safe_store_tm(
        self,
        tm_layer: TranslationMemory,
        chunk: TextChunk,
        translated_chunk: TranslatedChunk,
        glossary_hash: str,
        errors: List[str],
    ) -> None:
        """Store TM entry and continue on failures."""
        try:
            metadata = {
                "translator_used": translated_chunk.translator_used,
                "estimated_tokens": translated_chunk.estimated_tokens,
                "model": self.config.model,
                "translator_backend": self.config.translator_type,
            }
            if translated_chunk.metadata:
                metadata["backend_metadata"] = translated_chunk.metadata
            tm_layer.store_entry(
                source_text=chunk.original_text,
                translated_text=translated_chunk.translated_text,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                glossary_hash=glossary_hash,
                metadata=metadata,
            )
        except TranslationMemoryError as exc:
            message = f"TM store failed for chunk {chunk.index}: {exc}"
            logger.warning(message)
            errors.append(message)

    def _build_reused_chunk(
        self,
        chunk: TextChunk,
        translated_text: str,
        source_tag: str,
        metadata: Optional[dict] = None,
    ) -> TranslatedChunk:
        """Build a translated chunk object for cache/TM reuse."""
        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=translated_text,
            word_count=chunk.word_count,
            translator_used=f"{self.translator.get_name()}[{source_tag}]",
            translation_source=source_tag,
            metadata=metadata or {},
        )

    def _build_tm_chunk(
        self, chunk: TextChunk, tm_match: TranslationMemoryMatch
    ) -> TranslatedChunk:
        """Convert TM match into a TranslatedChunk."""
        estimated_tokens = None
        raw_estimate = tm_match.metadata.get("estimated_tokens")
        if isinstance(raw_estimate, int):
            estimated_tokens = raw_estimate
        else:
            usage = tm_match.metadata.get("backend_metadata", {})
            if isinstance(usage, dict):
                usage_payload = usage.get("usage", {})
                if isinstance(usage_payload, dict):
                    total = usage_payload.get("total_tokens")
                    if isinstance(total, int):
                        estimated_tokens = total

        return TranslatedChunk(
            index=chunk.index,
            original_text=chunk.original_text,
            translated_text=tm_match.translated_text,
            word_count=chunk.word_count,
            translator_used=f"{self.translator.get_name()}[tm]",
            translation_source="tm",
            estimated_tokens=estimated_tokens,
            metadata=tm_match.metadata,
        )

    def _cache_model_key(self) -> str:
        """Build cache model dimension key with backend disambiguation."""
        if self.config.translator_type == "openai":
            return self.config.model
        return self.translator.get_name()

    def _write_output(self, text: str) -> None:
        """Write translated text to output file."""
        try:
            with open(self.config.output_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            logger.info("Output written to: %s", self.config.output_path)
        except IOError as exc:
            logger.error("Failed to write output file: %s", exc)
            raise

    def _write_report(
        self,
        *,
        started_at: str,
        started_at_dt: datetime,
        total_chunks: int,
        translated_chunks: List[TranslatedChunk],
        cache_hits: int,
        cache_misses: int,
        tm_hits: int,
        errors: List[str],
    ) -> None:
        """Write JSON report when report_path is configured."""
        if self.config.report_path is None:
            return

        finished_at_dt = datetime.now(timezone.utc)
        finished_at = finished_at_dt.isoformat()
        elapsed_seconds = (finished_at_dt - started_at_dt).total_seconds()

        token_values = [
            chunk.estimated_tokens
            for chunk in translated_chunks
            if isinstance(chunk.estimated_tokens, int)
        ]
        estimated_token_usage = sum(token_values) if token_values else None

        report = TranslationRunReport(
            input_file=str(self.config.input_path),
            output_file=str(self.config.output_path),
            translator_backend=self.config.translator_type,
            model=self.config.model,
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            total_chunks=total_chunks,
            translated_chunks=len(translated_chunks),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            translation_memory_hits=tm_hits,
            estimated_token_usage=estimated_token_usage,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=round(elapsed_seconds, 4),
            errors=errors,
        )

        try:
            ReportWriter.write(report, self.config.report_path)
        except Exception as exc:
            logger.error("Failed to write report file: %s", exc, exc_info=True)
