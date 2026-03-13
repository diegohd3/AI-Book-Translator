"""
Main orchestration pipeline for the translation workflow.

Coordinates all modules to perform end-to-end document translation.
"""

import logging

from config import TranslationConfig
from txt_extractor import TxtExtractor
from cleaner import TextCleaner
from chunker import TextChunker
from translator import BaseTranslator, TranslatorFactory
from assembler import TranslationAssembler
from models import TranslationResult

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    End-to-end translation pipeline.
    
    Orchestrates the complete workflow:
    1. Extract text from file
    2. Clean text
    3. Chunk text by paragraphs
    4. Build translation prompts
    5. Translate each chunk
    6. Assemble translated chunks
    7. Write output file
    """

    def __init__(self, config: TranslationConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: TranslationConfig instance with all settings
        """
        self.config = config
        self.extractor = TxtExtractor(encoding="utf-8")
        self.cleaner = TextCleaner(preserve_paragraphs=True)
        self.chunker = TextChunker(target_chunk_size=config.chunk_size)
        self.assembler = TranslationAssembler()

        # Create translator dynamically based on config
        self.translator = self._create_translator()

    def _create_translator(self) -> BaseTranslator:
        """
        Create a translator instance based on configuration.
        
        Returns:
            A BaseTranslator instance
        """
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
            A TranslationResult containing all translated content and metadata
            
        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If any pipeline step fails
        """
        logger.info("=" * 60)
        logger.info("Starting translation pipeline")
        logger.info(f"Input: {self.config.input_path}")
        logger.info(f"Output: {self.config.output_path}")
        logger.info(f"Translator: {self.translator.get_name()}")
        logger.info("=" * 60)

        try:
            # Step 1: Extract text
            logger.info("Step 1: Extracting text from file...")
            original_text = self.extractor.extract(self.config.input_path)

            # Step 2: Clean text
            logger.info("Step 2: Cleaning text...")
            cleaned_text = self.cleaner.clean(original_text)

            # Step 3: Chunk text
            logger.info("Step 3: Chunking text...")
            chunks = self.chunker.chunk(cleaned_text)

            if not chunks:
                raise ValueError("No chunks were created from the text")

            logger.info(f"Created {len(chunks)} chunks")

            # Step 4-5: Translate each chunk
            logger.info("Step 4-5: Building prompts and translating chunks...")
            translated_chunks = []

            for i, chunk in enumerate(chunks):
                logger.info(f"  Translating chunk {i + 1}/{len(chunks)} ({chunk.word_count} words)...")
                translated_chunk = self.translator.translate(chunk)
                translated_chunks.append(translated_chunk)

            # Step 6: Assemble translated chunks
            logger.info("Step 6: Assembling translated chunks...")
            final_translated_text = self.assembler.assemble(translated_chunks)

            # Step 7: Write output file
            logger.info("Step 7: Writing output file...")
            self.config.ensure_output_dir_exists()
            self._write_output(final_translated_text)

            logger.info("=" * 60)
            logger.info("Translation pipeline completed successfully!")
            logger.info("=" * 60)

            # Create and return result object
            result = TranslationResult(
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                original_text=original_text,
                translated_text=final_translated_text,
                chunks=translated_chunks,
                total_chunks=len(chunks),
                translator_used=self.translator.get_name(),
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise

    def _write_output(self, text: str) -> None:
        """
        Write translated text to output file.
        
        Args:
            text: Text to write
        """
        try:
            with open(self.config.output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Output written to: {self.config.output_path}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
            raise
