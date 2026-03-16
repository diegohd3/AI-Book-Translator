"""
EPUB extraction module.

Builds a translation-ready text stream with stable segment placeholders while
keeping in-memory references to original XHTML nodes for round-trip assembly.
"""

from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from lxml import etree

logger = logging.getLogger(__name__)

_XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}
_SKIP_TEXT_TAGS = {
    "script",
    "style",
    "noscript",
    "code",
    "pre",
    "svg",
    "math",
}
_XML_DECLARATION_RE = re.compile(rb"^\s*<\?xml\b")


@dataclass
class ParsedXmlDocument:
    """Parsed XML document and serialization hints."""

    tree: etree._ElementTree
    encoding: str
    has_xml_declaration: bool
    doctype: Optional[str]


@dataclass
class EpubTextSlot:
    """
    Mutable reference to an XHTML text location.

    `slot` points to either `element.text` or `element.tail`.
    """

    segment_id: int
    element: etree._Element
    slot: str
    leading_whitespace: str
    core_text: str
    trailing_whitespace: str

    def set_translated_text(self, translated_core: str) -> None:
        """Apply translated core text while preserving original edge whitespace."""
        value = f"{self.leading_whitespace}{translated_core}{self.trailing_whitespace}"
        if self.slot == "text":
            self.element.text = value
            return
        self.element.tail = value


@dataclass
class EpubExtractionState:
    """Extraction state required for deterministic EPUB assembly."""

    zip_infos: List[zipfile.ZipInfo]
    archive_entries: Dict[str, bytes]
    opf_path: str
    opf_document: ParsedXmlDocument
    xhtml_documents: Dict[str, ParsedXmlDocument]
    segments: List[EpubTextSlot]


class EpubExtractor:
    """
    Extracts translatable text nodes from EPUB content documents.

    Segment payload format:
      [[TB_SEG_000001_START]]
      text to translate
      [[TB_SEG_000001_END]]
    """

    def __init__(self):
        self._state: Optional[EpubExtractionState] = None

    @property
    def state(self) -> EpubExtractionState:
        """Return extraction state after `extract()`."""
        if self._state is None:
            raise RuntimeError("EPUB extraction state is not initialized")
        return self._state

    @staticmethod
    def segment_start_marker(segment_id: int) -> str:
        return f"[[TB_SEG_{segment_id:06d}_START]]"

    @staticmethod
    def segment_end_marker(segment_id: int) -> str:
        return f"[[TB_SEG_{segment_id:06d}_END]]"

    def extract(self, file_path: Path) -> str:
        """
        Extract translatable text stream from an EPUB file.

        Returns:
            Combined marker-wrapped segments separated with blank lines.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info("Extracting translatable content from EPUB: %s", file_path)

        with zipfile.ZipFile(file_path, "r") as archive:
            zip_infos = list(archive.infolist())
            archive_entries = {
                info.filename: archive.read(info.filename) for info in zip_infos
            }

        if "META-INF/container.xml" not in archive_entries:
            raise ValueError("Invalid EPUB: META-INF/container.xml is missing")

        container_doc = self._parse_xml_document(archive_entries["META-INF/container.xml"])
        opf_path = self._discover_opf_path(container_doc.tree)

        if opf_path not in archive_entries:
            raise ValueError(f"Invalid EPUB: OPF package not found at '{opf_path}'")

        opf_document = self._parse_xml_document(archive_entries[opf_path])
        xhtml_paths = self._collect_xhtml_paths(opf_document.tree, opf_path)

        xhtml_documents: Dict[str, ParsedXmlDocument] = {}
        segments: List[EpubTextSlot] = []
        rendered_segments: List[str] = []
        segment_id = 1

        for xhtml_path in xhtml_paths:
            payload = archive_entries.get(xhtml_path)
            if payload is None:
                logger.warning("Manifest item not found in archive: %s", xhtml_path)
                continue

            document = self._parse_xml_document(payload)
            xhtml_documents[xhtml_path] = document

            body = self._find_body(document.tree.getroot())
            if body is None:
                continue

            for element, slot_name, raw_text in self._iter_translatable_slots(body):
                leading, core, trailing = self._split_edge_whitespace(raw_text)
                if not core:
                    continue

                segments.append(
                    EpubTextSlot(
                        segment_id=segment_id,
                        element=element,
                        slot=slot_name,
                        leading_whitespace=leading,
                        core_text=core,
                        trailing_whitespace=trailing,
                    )
                )
                rendered_segments.append(
                    self._render_segment_payload(segment_id=segment_id, text=core)
                )
                segment_id += 1

        if not rendered_segments:
            raise ValueError("No translatable text content was found in EPUB")

        self._state = EpubExtractionState(
            zip_infos=zip_infos,
            archive_entries=archive_entries,
            opf_path=opf_path,
            opf_document=opf_document,
            xhtml_documents=xhtml_documents,
            segments=segments,
        )

        logger.info(
            "EPUB extraction complete: %s XHTML documents, %s text segments",
            len(xhtml_documents),
            len(segments),
        )
        return "\n\n".join(rendered_segments)

    def _render_segment_payload(self, *, segment_id: int, text: str) -> str:
        start = self.segment_start_marker(segment_id)
        end = self.segment_end_marker(segment_id)
        return f"{start}\n{text}\n{end}"

    def _discover_opf_path(self, container_tree: etree._ElementTree) -> str:
        rootfiles = container_tree.xpath(
            "/*[local-name()='container']/*[local-name()='rootfiles']/*[local-name()='rootfile']"
        )
        if not rootfiles:
            raise ValueError("Invalid EPUB: no rootfile declared in container.xml")

        opf_path = str(rootfiles[0].get("full-path", "")).strip()
        if not opf_path:
            raise ValueError("Invalid EPUB: rootfile is missing full-path")
        return posixpath.normpath(opf_path)

    def _collect_xhtml_paths(self, opf_tree: etree._ElementTree, opf_path: str) -> List[str]:
        manifest_items = opf_tree.xpath(
            "/*[local-name()='package']/*[local-name()='manifest']/*[local-name()='item']"
        )
        item_by_id = {}
        ordered_items = []
        for item in manifest_items:
            item_id = str(item.get("id", "")).strip()
            href = str(item.get("href", "")).strip()
            media_type = str(item.get("media-type", "")).strip().lower()
            if not href:
                continue
            resolved_path = self._resolve_opf_href(opf_path, href)
            manifest_record = {
                "id": item_id,
                "href": href,
                "resolved_path": resolved_path,
                "media_type": media_type,
            }
            ordered_items.append(manifest_record)
            if item_id:
                item_by_id[item_id] = manifest_record

        spine_paths: List[str] = []
        for itemref in opf_tree.xpath(
            "/*[local-name()='package']/*[local-name()='spine']/*[local-name()='itemref']"
        ):
            idref = str(itemref.get("idref", "")).strip()
            if not idref:
                continue
            manifest_item = item_by_id.get(idref)
            if not manifest_item:
                continue
            if manifest_item["media_type"] in _XHTML_MEDIA_TYPES:
                spine_paths.append(manifest_item["resolved_path"])

        remaining_paths: List[str] = []
        for item in ordered_items:
            if item["media_type"] not in _XHTML_MEDIA_TYPES:
                continue
            resolved_path = item["resolved_path"]
            if resolved_path in spine_paths or resolved_path in remaining_paths:
                continue
            remaining_paths.append(resolved_path)

        return spine_paths + remaining_paths

    @staticmethod
    def _resolve_opf_href(opf_path: str, href: str) -> str:
        href_without_fragment = href.split("#", 1)[0]
        opf_dir = posixpath.dirname(opf_path)
        if opf_dir:
            return posixpath.normpath(posixpath.join(opf_dir, href_without_fragment))
        return posixpath.normpath(href_without_fragment)

    def _iter_translatable_slots(
        self,
        node: etree._Element,
        *,
        skip_subtree: bool = False,
    ) -> Iterable[tuple[etree._Element, str, str]]:
        local_name = self._local_name(node.tag)
        should_skip = skip_subtree or local_name in _SKIP_TEXT_TAGS

        if not should_skip and node.text and node.text.strip():
            yield (node, "text", node.text)

        for child in node:
            if isinstance(child.tag, str):
                yield from self._iter_translatable_slots(child, skip_subtree=should_skip)
            if not should_skip and child.tail and child.tail.strip():
                yield (child, "tail", child.tail)

    @staticmethod
    def _split_edge_whitespace(value: str) -> tuple[str, str, str]:
        leading_match = re.match(r"^\s*", value)
        trailing_match = re.search(r"\s*$", value)

        leading = leading_match.group(0) if leading_match else ""
        trailing = trailing_match.group(0) if trailing_match else ""

        start_index = len(leading)
        end_index = len(value) - len(trailing)
        if end_index < start_index:
            end_index = start_index
        core = value[start_index:end_index]
        return leading, core, trailing

    def _find_body(self, root: etree._Element) -> Optional[etree._Element]:
        if self._local_name(root.tag) == "body":
            return root
        matches = root.xpath("//*[local-name()='body']")
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _parse_xml_document(payload: bytes) -> ParsedXmlDocument:
        parser = etree.XMLParser(
            recover=True,
            remove_blank_text=False,
            resolve_entities=False,
            huge_tree=True,
        )
        tree = etree.parse(BytesIO(payload), parser)
        docinfo = tree.docinfo
        encoding = docinfo.encoding or "utf-8"
        has_xml_declaration = bool(_XML_DECLARATION_RE.match(payload))
        doctype = docinfo.doctype if docinfo.doctype else None
        return ParsedXmlDocument(
            tree=tree,
            encoding=encoding,
            has_xml_declaration=has_xml_declaration,
            doctype=doctype,
        )

    @staticmethod
    def _local_name(tag_value: object) -> str:
        if not isinstance(tag_value, str):
            return ""
        if "}" in tag_value:
            return tag_value.rsplit("}", 1)[-1].lower()
        return tag_value.lower()
