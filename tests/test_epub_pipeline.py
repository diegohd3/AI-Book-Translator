import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from lxml import etree

from config import TranslationConfig
from pipeline import TranslationPipeline


class EpubPipelineTests(unittest.TestCase):
    def test_epub_roundtrip_preserves_resources_and_updates_language_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_epub = tmp_path / "input.epub"
            output_epub = tmp_path / "output.epub"
            self._create_sample_epub(input_epub)

            config = TranslationConfig(
                input_path=input_epub,
                output_path=output_epub,
                source_language="en",
                target_language="es",
                translator_type="mock",
                cache_db_path=tmp_path / "cache.sqlite3",
                report_output_path=tmp_path / "output.report.json",
            )

            TranslationPipeline(config).run()

            self.assertTrue(output_epub.exists())

            with zipfile.ZipFile(input_epub, "r") as original_archive, zipfile.ZipFile(
                output_epub,
                "r",
            ) as translated_archive:
                original_names = sorted(original_archive.namelist())
                translated_names = sorted(translated_archive.namelist())
                self.assertEqual(original_names, translated_names)

                resource_names = [
                    name
                    for name in original_names
                    if name.endswith(".css")
                    or name.endswith(".jpg")
                    or name.endswith(".png")
                    or name.endswith(".ttf")
                ]
                self.assertGreater(len(resource_names), 0)

                for name in resource_names:
                    original_hash = hashlib.sha256(original_archive.read(name)).hexdigest()
                    translated_hash = hashlib.sha256(translated_archive.read(name)).hexdigest()
                    self.assertEqual(original_hash, translated_hash, msg=name)

                opf_root = etree.fromstring(translated_archive.read("OEBPS/content.opf"))
                language = opf_root.xpath(
                    "string(/*[local-name()='package']/*[local-name()='metadata']/*[local-name()='language'][1])"
                )
                self.assertEqual("es", language.strip())

                title = opf_root.xpath(
                    "string(/*[local-name()='package']/*[local-name()='metadata']/*[local-name()='title'][1])"
                )
                identifier = opf_root.xpath(
                    "string(/*[local-name()='package']/*[local-name()='metadata']/*[local-name()='identifier'][1])"
                )
                self.assertEqual("Test Book", title.strip())
                self.assertEqual("urn:uuid:test-book", identifier.strip())

    def test_epub_translation_preserves_xhtml_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_epub = tmp_path / "input.epub"
            output_epub = tmp_path / "output.epub"
            self._create_sample_epub(input_epub)

            config = TranslationConfig(
                input_path=input_epub,
                output_path=output_epub,
                source_language="en",
                target_language="es",
                translator_type="mock",
                cache_db_path=tmp_path / "cache.sqlite3",
                report_output_path=tmp_path / "output.report.json",
            )

            TranslationPipeline(config).run()

            with zipfile.ZipFile(output_epub, "r") as translated_archive:
                chapter_payload = translated_archive.read("OEBPS/chapter1.xhtml")
                chapter_text = chapter_payload.decode("utf-8")
                self.assertNotIn("[[TB_SEG_", chapter_text)

                chapter_root = etree.fromstring(chapter_payload)
                emphasis_nodes = chapter_root.xpath(
                    "//*[local-name()='em' and @class='spell']"
                )
                self.assertEqual(1, len(emphasis_nodes))

                paragraph_nodes = chapter_root.xpath(
                    "//*[local-name()='p' and @id='p1']"
                )
                self.assertEqual(1, len(paragraph_nodes))
                paragraph_text = "".join(paragraph_nodes[0].itertext())
                self.assertIn("el/la", paragraph_text)

                nav_root = etree.fromstring(translated_archive.read("OEBPS/nav.xhtml"))
                href_values = nav_root.xpath("//*[local-name()='a']/@href")
                self.assertIn("chapter1.xhtml", href_values)

    def _create_sample_epub(self, output_path: Path) -> None:
        container_xml = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
        opf_xml = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:test-book</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="css" href="styles/book.css" media-type="text/css"/>
    <item id="img" href="images/cover.jpg" media-type="image/jpeg"/>
    <item id="font" href="fonts/book.ttf" media-type="font/ttf"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
    <itemref idref="nav"/>
  </spine>
</package>
"""
        chapter_xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Chapter 1</title>
    <link rel="stylesheet" type="text/css" href="styles/book.css"/>
  </head>
  <body>
    <h1 class="title">the beginning</h1>
    <p id="p1">the hero and the <em class="spell">mage</em> walked forward.</p>
    <p>This is the second paragraph.</p>
  </body>
</html>
"""
        nav_xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Navigation</title></head>
  <body>
    <nav epub:type="toc" xmlns:epub="http://www.idpf.org/2007/ops">
      <ol>
        <li><a href="chapter1.xhtml">Chapter One</a></li>
      </ol>
    </nav>
  </body>
</html>
"""
        css_payload = b"body { font-family: serif; color: #222; }\n"
        image_payload = b"\xff\xd8\xff\xe0TESTJPEGPAYLOAD\xff\xd9"
        font_payload = b"\x00\x01\x00\x00DUMMYFONTDATA"

        with zipfile.ZipFile(output_path, "w") as archive:
            mimetype_info = zipfile.ZipInfo("mimetype")
            mimetype_info.compress_type = zipfile.ZIP_STORED
            archive.writestr(mimetype_info, b"application/epub+zip")
            archive.writestr("META-INF/container.xml", container_xml.encode("utf-8"))
            archive.writestr("OEBPS/content.opf", opf_xml.encode("utf-8"))
            archive.writestr("OEBPS/chapter1.xhtml", chapter_xhtml.encode("utf-8"))
            archive.writestr("OEBPS/nav.xhtml", nav_xhtml.encode("utf-8"))
            archive.writestr("OEBPS/styles/book.css", css_payload)
            archive.writestr("OEBPS/images/cover.jpg", image_payload)
            archive.writestr("OEBPS/fonts/book.ttf", font_payload)


if __name__ == "__main__":
    unittest.main()
