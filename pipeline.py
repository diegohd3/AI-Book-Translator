"""
Main orchestration pipeline for the translation workflow.

Coordinates all modules to perform end-to-end document translation.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

from assembler import TranslationAssembler
from cache import CacheError, TranslationCache
from chunk_signals import analyze_chunk_signal
from chunker import TextChunker
from cleaner import TextCleaner
from config import TranslationConfig
from context_memory import BookContextMemory
from document_format import detect_document_format
from glossary import (
    Glossary,
    GlossaryMatch,
    GlossaryValidationError,
    detect_missing_forced_terms,
    find_relevant_terms,
    find_used_glossary_terms,
    load_glossary,
)
from models import TextChunk, TranslatedChunk, TranslationResult
from normalization import sha256_hexdigest
from reporting import ChunkReportEntry, ReportWriter, TranslationRunReport
from prompt_builder import get_prompt_policy_version
from style_profile import StyleProfile, default_style_profile, load_style_profile
from translation_memory import (
    TranslationMemory,
    TranslationMemoryError,
    TranslationMemoryMatch,
)
from translator import BaseTranslator, TranslatorFactory
from txt_extractor import TxtExtractor

logger = logging.getLogger(__name__)

_UNCERTAIN_RE = re.compile(r"\?\?|\[uncertain\]|\(unclear\)", flags=re.IGNORECASE)
_ASCII_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_EPUB_SEGMENT_MARKER_RE = re.compile(r"\[\[TB_SEG_\d{6}_(?:START|END)\]\]")
_ENGLISH_HINT_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "without",
    "this",
    "that",
    "were",
    "was",
    "into",
    "through",
    "because",
    "would",
}
_SPANISH_COMMA_BEFORE_CONJ_RE = re.compile(r",\s+(?:y|o|e|u)\b", flags=re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+[,:;!?]")
_DOUBLE_COMMA_RE = re.compile(r",\s*,")


class TranslationPipeline:
    """End-to-end translation pipeline with literary intelligence features."""

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.cleaner = TextCleaner(preserve_paragraphs=True)
        self.chunker = TextChunker(target_chunk_size=config.chunk_size)
        self.document_format = detect_document_format(
            self.config.input_path,
            self.config.input_format,
        )
        self.extractor, self.assembler, self._requires_cleaning = self._build_document_io()
        self.translator = self._create_translator()

    def _build_document_io(self):
        """Create extractor/assembler pair based on input format."""
        if self.document_format == "txt":
            return TxtExtractor(encoding="utf-8"), TranslationAssembler(), True

        if self.document_format == "epub":
            from epub_assembler import EpubAssembler
            from epub_extractor import EpubExtractor

            epub_extractor = EpubExtractor()
            return epub_extractor, EpubAssembler(epub_extractor), False

        raise ValueError(f"Unsupported document format: {self.document_format}")

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
        """Execute the complete translation pipeline."""
        started_at_dt = datetime.now(timezone.utc)
        started_at = started_at_dt.isoformat()
        errors: List[str] = []

        total_chunks = 0
        cache_hits = 0
        cache_misses = 0
        tm_hits = 0
        original_text = ""
        translated_chunks: List[TranslatedChunk] = []

        chunk_reports: List[ChunkReportEntry] = []
        glossary_relevant_counter: Dict[str, int] = {}
        glossary_used_counter: Dict[str, int] = {}
        consistency_risks: List[str] = []
        uncertain_translations: List[str] = []

        style_profile = default_style_profile(
            source_language=self.config.source_language,
            target_language=self.config.target_language,
        )
        style_profile_hash = style_profile.profile_hash()
        policy_hash = ""
        glossary_hash = "no_glossary"
        glossary_total_terms = 0
        recurring_terms: Dict[str, List[str]] = {}

        logger.info("=" * 60)
        logger.info("Starting literary translation pipeline")
        logger.info("Input: %s", self.config.input_path)
        logger.info("Output: %s", self.config.output_path)
        logger.info("Detected format: %s", self.document_format)
        logger.info("Report: %s", self.config.report_output_path)
        logger.info("Translator: %s", self.translator.get_name())
        logger.info("Refinement enabled: %s", self.config.enable_refinement)
        logger.info("=" * 60)

        cache_layer: Optional[TranslationCache] = None
        tm_layer: Optional[TranslationMemory] = None
        context_memory: Optional[BookContextMemory] = None

        try:
            style_profile = self._load_style_profile()
            style_profile_hash = style_profile.profile_hash()
            self.translator.configure_style_profile(style_profile)

            glossary = self._load_glossary()
            glossary_hash = glossary.glossary_hash()
            glossary_total_terms = len(glossary.terms)
            self.translator.configure_glossary(glossary.terms)

            policy_hash = self._build_policy_hash(
                style_profile_hash=style_profile_hash,
                glossary_hash=glossary_hash,
            )

            context_memory = BookContextMemory(style_profile)

            cache_layer = self._initialize_cache(errors)
            tm_layer = self._initialize_translation_memory(errors)

            logger.info("Step 1: Extracting text from file...")
            original_text = self.extractor.extract(self.config.input_path)

            if self._requires_cleaning:
                logger.info("Step 2: Cleaning text...")
                cleaned_text = self.cleaner.clean(original_text)
            else:
                logger.info("Step 2: Cleaning skipped for format '%s'", self.document_format)
                cleaned_text = original_text

            logger.info("Step 3: Chunking text...")
            chunks = self.chunker.chunk(cleaned_text)
            if not chunks:
                raise ValueError("No chunks were created from the text")
            total_chunks = len(chunks)
            logger.info("Created %s chunks", total_chunks)

            logger.info("Step 4-6: Reusing and translating chunks...")
            for sequence_index, chunk in enumerate(chunks, start=1):
                logger.info(
                    "  Processing chunk %s/%s (%s words)...",
                    sequence_index,
                    total_chunks,
                    chunk.word_count,
                )

                source_text_for_analysis = self._strip_epub_segment_markers(
                    chunk.original_text
                )
                chunk_signal = analyze_chunk_signal(source_text_for_analysis)
                relevant_matches = find_relevant_terms(
                    source_text_for_analysis,
                    glossary.terms,
                )

                snapshot = {}
                if context_memory is not None:
                    snapshot = context_memory.prompt_snapshot(self.config.context_window)

                prompt_metadata = {
                    "chunk_signal": chunk_signal.label,
                    "context_snapshot": snapshot,
                    "relevant_glossary_terms": relevant_matches,
                }

                translated = None
                stage1_translator = self.translator.get_name()
                stage2_translator: Optional[str] = None
                refinement_applied = False
                refinement_drift: Optional[float] = None

                if cache_layer is not None:
                    try:
                        cached_text = cache_layer.get_cached_translation(
                            source_text=chunk.original_text,
                            source_language=self.config.source_language,
                            target_language=self.config.target_language,
                            model=self._cache_model_key(policy_hash),
                            glossary_hash=policy_hash,
                        )
                        if cached_text is not None:
                            cache_hits += 1
                            translated = self._build_reused_chunk(
                                chunk=chunk,
                                translated_text=cached_text,
                                source_tag="cache",
                                metadata={
                                    "policy_hash": policy_hash,
                                    "chunk_signal": chunk_signal.label,
                                },
                            )
                            stage1_translator = translated.translator_used
                            refinement_applied = (
                                self.config.enable_refinement
                                and self.translator.supports_refinement()
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
                            glossary_hash=policy_hash,
                        )
                        if tm_match is not None:
                            tm_hits += 1
                            translated = self._build_tm_chunk(chunk, tm_match)
                            stage1_translator = translated.translator_used
                            refinement_applied = (
                                self.config.enable_refinement
                                and self.translator.supports_refinement()
                            )
                            if cache_layer is not None:
                                self._safe_store_cache(
                                    cache_layer=cache_layer,
                                    chunk=chunk,
                                    translated_text=translated.translated_text,
                                    policy_hash=policy_hash,
                                    errors=errors,
                                )
                    except TranslationMemoryError as exc:
                        message = (
                            f"Translation memory lookup failed for chunk {chunk.index}: {exc}"
                        )
                        logger.warning(message)
                        errors.append(message)

                if translated is None:
                    stage1_chunk = self.translator.translate(
                        chunk,
                        prompt_metadata=prompt_metadata,
                    )
                    stage1_translator = stage1_chunk.translator_used
                    translated = stage1_chunk
                    translated.translation_source = "translator"

                    if self.config.enable_refinement:
                        stage2_chunk = self.translator.refine(
                            chunk,
                            stage1_chunk.translated_text,
                            prompt_metadata=prompt_metadata,
                        )
                        stage2_translator = stage2_chunk.translator_used
                        refinement_applied = self.translator.supports_refinement()
                        refinement_drift = self._calculate_refinement_drift(
                            stage1_chunk.translated_text,
                            stage2_chunk.translated_text,
                        )

                        translated = TranslatedChunk(
                            index=stage2_chunk.index,
                            original_text=stage2_chunk.original_text,
                            translated_text=stage2_chunk.translated_text,
                            word_count=stage2_chunk.word_count,
                            translator_used=stage2_chunk.translator_used,
                            translation_source="translator",
                            estimated_tokens=stage2_chunk.estimated_tokens
                            or stage1_chunk.estimated_tokens,
                            metadata={
                                "stage1_translator": stage1_translator,
                                "stage2_translator": stage2_translator,
                                "refinement_enabled": True,
                                "refinement_applied": refinement_applied,
                                "refinement_drift": refinement_drift,
                                "chunk_signal": chunk_signal.label,
                                "stage1_metadata": stage1_chunk.metadata,
                                "stage2_metadata": stage2_chunk.metadata,
                            },
                        )
                    else:
                        translated.metadata = {
                            **translated.metadata,
                            "stage1_translator": stage1_translator,
                            "stage2_translator": None,
                            "refinement_enabled": False,
                            "refinement_applied": False,
                            "refinement_drift": None,
                            "chunk_signal": chunk_signal.label,
                        }

                    if cache_layer is not None:
                        self._safe_store_cache(
                            cache_layer=cache_layer,
                            chunk=chunk,
                            translated_text=translated.translated_text,
                            policy_hash=policy_hash,
                            errors=errors,
                        )
                    if tm_layer is not None:
                        self._safe_store_tm(
                            tm_layer=tm_layer,
                            chunk=chunk,
                            translated_chunk=translated,
                            policy_hash=policy_hash,
                            errors=errors,
                        )

                relevant_sources = [match.term.source for match in relevant_matches]
                translated_text_for_analysis = self._strip_epub_segment_markers(
                    translated.translated_text
                )
                used_targets = find_used_glossary_terms(
                    translated_text_for_analysis,
                    relevant_matches,
                )
                used_lookup = {item.casefold() for item in used_targets}
                used_sources = [
                    match.term.source
                    for match in relevant_matches
                    if match.term.target.casefold() in used_lookup
                ]

                warnings = self._build_chunk_warnings(
                    translated_text=translated_text_for_analysis,
                    relevant_matches=relevant_matches,
                    used_targets=used_targets,
                    refinement_drift=refinement_drift,
                    translator_used=translated.translator_used,
                )

                if context_memory is not None:
                    context_memory.update_from_chunk(
                        chunk_index=chunk.index,
                        source_text=source_text_for_analysis,
                        translated_text=translated_text_for_analysis,
                        chunk_signal=chunk_signal.label,
                        glossary_matches=relevant_matches,
                        glossary_used=used_targets,
                        warnings=warnings,
                    )

                self._update_glossary_counters(
                    glossary_relevant_counter,
                    glossary_used_counter,
                    relevant_matches,
                    used_targets,
                )

                for warning in warnings:
                    if warning.startswith("UNCERTAIN"):
                        uncertain_translations.append(f"chunk {chunk.index}: {warning}")
                    if warning.startswith("RISK"):
                        consistency_risks.append(f"chunk {chunk.index}: {warning}")

                chunk_reports.append(
                    ChunkReportEntry(
                        index=chunk.index,
                        word_count=chunk.word_count,
                        chunk_signal=chunk_signal.label,
                        translation_source=translated.translation_source,
                        refinement_enabled=self.config.enable_refinement,
                        refinement_applied=refinement_applied,
                        stage1_translator=stage1_translator,
                        stage2_translator=stage2_translator,
                        glossary_relevant=sorted(set(relevant_sources)),
                        glossary_used=sorted(set(used_sources)),
                        warnings=warnings,
                        diagnostics={
                            "dialogue_score": chunk_signal.dialogue_score,
                            "emphasis_score": chunk_signal.emphasis_score,
                            "refinement_drift": refinement_drift,
                        },
                    )
                )

                translated_chunks.append(translated)

            logger.info("Step 7: Assembling translated chunks...")
            final_translated_text = self.assembler.assemble(translated_chunks)

            logger.info("Step 8: Writing output file...")
            self.config.ensure_output_dir_exists()
            self._write_output(final_translated_text)

            if context_memory is not None:
                recurring_terms = context_memory.recurring_terms_summary()
                consistency_risks.extend(context_memory.consistency_risks())

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
                style_profile=style_profile,
                style_profile_hash=style_profile_hash,
                policy_hash=policy_hash,
                glossary_total_terms=glossary_total_terms,
                glossary_relevant_counter=glossary_relevant_counter,
                glossary_used_counter=glossary_used_counter,
                chunk_reports=chunk_reports,
                recurring_terms=recurring_terms,
                consistency_risks=consistency_risks,
                uncertain_translations=uncertain_translations,
            )

    def _load_style_profile(self) -> StyleProfile:
        """Load optional style profile or return default profile."""
        if self.config.style_profile_path is None:
            logger.info("No style profile provided; using default literary profile")
            return default_style_profile(
                source_language=self.config.source_language,
                target_language=self.config.target_language,
            )

        logger.info("Loading style profile from: %s", self.config.style_profile_path)
        try:
            return load_style_profile(self.config.style_profile_path)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"Invalid style profile configuration: {exc}") from exc

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
        policy_hash: str,
        errors: List[str],
    ) -> None:
        """Store cache entry and continue on failures."""
        try:
            cache_layer.store_cached_translation(
                source_text=chunk.original_text,
                translated_text=translated_text,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                model=self._cache_model_key(policy_hash),
                glossary_hash=policy_hash,
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
        policy_hash: str,
        errors: List[str],
    ) -> None:
        """Store TM entry and continue on failures."""
        try:
            metadata = {
                "translator_used": translated_chunk.translator_used,
                "estimated_tokens": translated_chunk.estimated_tokens,
                "model": self.config.model,
                "translator_backend": self.config.translator_type,
                "chunk_metadata": translated_chunk.metadata,
            }
            tm_layer.store_entry(
                source_text=chunk.original_text,
                translated_text=translated_chunk.translated_text,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                glossary_hash=policy_hash,
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

    def _cache_model_key(self, policy_hash: str) -> str:
        """Build cache model key partitioned by policy hash."""
        if self.config.translator_type == "openai":
            base_model = self.config.model
        else:
            base_model = self.translator.get_name()
        return f"{base_model}|policy:{policy_hash}"

    def _build_policy_hash(self, *, style_profile_hash: str, glossary_hash: str) -> str:
        """Compute policy hash used to partition cache/TM reuse."""
        model_dimension = (
            self.config.model.strip()
            if self.config.translator_type == "openai"
            else self.translator.get_name()
        )
        payload = (
            f"style:{style_profile_hash}|glossary:{glossary_hash}|"
            f"refinement:{self.config.enable_refinement}|context:{self.config.context_window}|"
            f"format:{self.document_format}|backend:{self.config.translator_type}|"
            f"model:{model_dimension}|prompt:{get_prompt_policy_version()}"
        )
        return sha256_hexdigest(payload)

    @staticmethod
    def _calculate_refinement_drift(stage1_text: str, stage2_text: str) -> float:
        """Return normalized edit drift between stage outputs."""
        ratio = SequenceMatcher(a=stage1_text or "", b=stage2_text or "").ratio()
        return round(1.0 - ratio, 4)

    def _build_chunk_warnings(
        self,
        *,
        translated_text: str,
        relevant_matches: Sequence[GlossaryMatch],
        used_targets: Sequence[str],
        refinement_drift: Optional[float],
        translator_used: Optional[str] = None,
    ) -> List[str]:
        warnings: List[str] = []

        missing_forced = detect_missing_forced_terms(relevant_matches, used_targets)
        for term in missing_forced:
            warnings.append(
                f"RISK: forced glossary term missing for '{term.source}' -> '{term.target}'"
            )

        if refinement_drift is not None and refinement_drift > 0.55:
            warnings.append(
                f"RISK: large refinement drift ({refinement_drift}) may indicate meaning shift"
            )

        translator_label = (translator_used or "").strip().lower()
        is_mock_translation = translator_label.startswith("mocktranslator")
        if not is_mock_translation:
            untranslated_spans = self._find_untranslated_english_spans(translated_text)
            if untranslated_spans:
                warnings.append(
                    "RISK: possible untranslated English span(s): "
                    + " | ".join(untranslated_spans[:2])
                )
            style_risks = self._find_spanish_style_risks(translated_text)
            for risk in style_risks:
                warnings.append(f"RISK: {risk}")

        if _UNCERTAIN_RE.search(translated_text):
            warnings.append("UNCERTAIN: translation contains uncertainty markers")

        if not translated_text.strip():
            warnings.append("UNCERTAIN: empty translated output")

        return warnings

    @staticmethod
    def _find_spanish_style_risks(text: str) -> List[str]:
        risks: List[str] = []
        value = text or ""

        if _SPANISH_COMMA_BEFORE_CONJ_RE.search(value):
            risks.append(
                "possible comma calque before a coordinating conjunction (y/o/e/u)"
            )
        if _SPACE_BEFORE_PUNCT_RE.search(value):
            risks.append("unexpected spacing before punctuation")
        if _DOUBLE_COMMA_RE.search(value):
            risks.append("repeated comma punctuation")
        if "?" in value and "¿" not in value:
            risks.append("question mark without opening inverted mark")
        if "!" in value and "¡" not in value:
            risks.append("exclamation mark without opening inverted mark")

        unique_risks: List[str] = []
        seen = set()
        for risk in risks:
            key = risk.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_risks.append(risk)
        return unique_risks

    @staticmethod
    def _find_untranslated_english_spans(text: str) -> List[str]:
        words = _ASCII_WORD_RE.findall(text or "")
        if len(words) < 4:
            return []

        spans: List[str] = []
        run: List[str] = []
        for token in words:
            run.append(token)
            if len(run) >= 4:
                lower_run = {item.lower() for item in run[-6:]}
                if lower_run.intersection(_ENGLISH_HINT_WORDS):
                    spans.append(" ".join(run[-6:]))

        unique: List[str] = []
        seen = set()
        for span in spans:
            key = span.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(span)
        return unique

    @staticmethod
    def _update_glossary_counters(
        relevant_counter: Dict[str, int],
        used_counter: Dict[str, int],
        relevant_matches: Sequence[GlossaryMatch],
        used_targets: Sequence[str],
    ) -> None:
        used_lookup = {target.casefold() for target in used_targets}
        for match in relevant_matches:
            key = f"{match.term.source} -> {match.term.target}"
            relevant_counter[key] = relevant_counter.get(key, 0) + match.count
            if match.term.target.casefold() in used_lookup:
                used_counter[key] = used_counter.get(key, 0) + 1

    def _strip_epub_segment_markers(self, text: str) -> str:
        """Remove marker placeholders used for EPUB segment round-tripping."""
        if self.document_format != "epub":
            return text

        stripped = _EPUB_SEGMENT_MARKER_RE.sub("", text)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    def _write_output(self, text: str) -> None:
        """Write translated output using format-specific assembler."""
        try:
            self.assembler.write_output(
                self.config.output_path,
                text,
                self.config.target_language,
            )
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
        style_profile: StyleProfile,
        style_profile_hash: str,
        policy_hash: str,
        glossary_total_terms: int,
        glossary_relevant_counter: Dict[str, int],
        glossary_used_counter: Dict[str, int],
        chunk_reports: List[ChunkReportEntry],
        recurring_terms: Dict[str, List[str]],
        consistency_risks: Sequence[str],
        uncertain_translations: Sequence[str],
    ) -> None:
        """Write JSON report to configured sidecar path."""
        if self.config.report_output_path is None:
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
            errors=list(errors),
            style_profile_hash=style_profile_hash,
            style_profile=style_profile.as_canonical_dict(),
            policy_hash=policy_hash,
            glossary_total_terms=glossary_total_terms,
            glossary_matches_relevant=dict(sorted(glossary_relevant_counter.items())),
            glossary_matches_used=dict(sorted(glossary_used_counter.items())),
            recurring_terms=recurring_terms,
            consistency_risks=sorted(set(consistency_risks)),
            uncertain_translations=sorted(set(uncertain_translations)),
            chunks=chunk_reports,
        )

        try:
            self.config.ensure_report_output_dir_exists()
            ReportWriter.write(report, self.config.report_output_path)
        except Exception as exc:
            logger.error("Failed to write report file: %s", exc, exc_info=True)
