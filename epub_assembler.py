"""
EPUB assembler module.

Applies translated text back into XHTML nodes and writes a new EPUB while
preserving non-text resources unchanged.
"""

from __future__ import annotations

import copy
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List

from lxml import etree

from assembler import TranslationAssembler
from epub_extractor import EpubExtractionState, EpubExtractor, EpubTextSlot
from models import TranslatedChunk

logger = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(
    r"\[\[TB_SEG_(\d{6})_START\]\]\s*\n?(.*?)\n?\s*\[\[TB_SEG_\1_END\]\]",
    flags=re.DOTALL,
)
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"


class EpubAssembler:
    """Assembles translated chunks and writes translated EPUB output."""

    def __init__(self, extractor: EpubExtractor):
        self.extractor = extractor
        self._text_assembler = TranslationAssembler()

    def assemble(self, chunks: List[TranslatedChunk]) -> str:
        """Assemble translated chunk text into a marker-wrapped stream."""
        return self._text_assembler.assemble(chunks)

    def write_output(self, output_path: Path, text: str, target_language: str) -> None:
        """Write translated EPUB preserving binary resources."""
        state = self.extractor.state
        translations = self._parse_segment_translations(text)
        self._apply_translations(state.segments, translations)
        self._update_language_metadata(state, target_language=target_language)
        self._write_archive(output_path, state)
        logger.info("EPUB output written to: %s", output_path)

    def _parse_segment_translations(self, text: str) -> Dict[int, str]:
        translations: Dict[int, str] = {}
        for match in _SEGMENT_RE.finditer(text):
            segment_id = int(match.group(1))
            translated_core = match.group(2).strip()
            if segment_id not in translations:
                translations[segment_id] = translated_core
        return translations

    def _apply_translations(
        self,
        segments: List[EpubTextSlot],
        translations: Dict[int, str],
    ) -> None:
        missing = 0
        for slot in segments:
            translated_core = translations.get(slot.segment_id)
            if translated_core is None:
                missing += 1
                translated_core = slot.core_text
            if not translated_core and slot.core_text:
                translated_core = slot.core_text
            slot.set_translated_text(translated_core)

        if missing:
            logger.warning(
                "Missing translated placeholders for %s segment(s); source text was retained.",
                missing,
            )

    def _update_language_metadata(
        self,
        state: EpubExtractionState,
        *,
        target_language: str,
    ) -> None:
        root = state.opf_document.tree.getroot()
        metadata_nodes = root.xpath("/*[local-name()='package']/*[local-name()='metadata']")
        if not metadata_nodes:
            return

        metadata = metadata_nodes[0]
        language_nodes = metadata.xpath("*[local-name()='language']")

        if language_nodes:
            language_nodes[0].text = target_language
            return

        dc_namespace = root.nsmap.get("dc", _DC_NAMESPACE)
        language_node = etree.Element(f"{{{dc_namespace}}}language")
        language_node.text = target_language
        metadata.append(language_node)

    def _write_archive(self, output_path: Path, state: EpubExtractionState) -> None:
        updated_entries: Dict[str, bytes] = dict(state.archive_entries)

        for path, document in state.xhtml_documents.items():
            updated_entries[path] = self._serialize_xml_document(document)
        updated_entries[state.opf_path] = self._serialize_xml_document(state.opf_document)

        with zipfile.ZipFile(output_path, "w") as output_archive:
            for original_info in state.zip_infos:
                filename = original_info.filename
                payload = updated_entries.get(filename)
                if payload is None:
                    continue

                write_info = copy.copy(original_info)
                output_archive.writestr(write_info, payload)

    @staticmethod
    def _serialize_xml_document(document) -> bytes:
        payload = etree.tostring(
            document.tree,
            encoding=document.encoding,
            xml_declaration=document.has_xml_declaration,
            doctype=document.doctype,
        )
        if isinstance(payload, str):
            return payload.encode(document.encoding or "utf-8")
        return payload
