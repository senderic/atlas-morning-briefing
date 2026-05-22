# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Comprehensive tests for epub_generator (XML escape, BMP filtering, main())."""

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.epub_generator import EPUBGenerator, main


class TestXmlEscape:
    def test_escapes_amp(self):
        assert EPUBGenerator._xml_escape("A & B") == "A &amp; B"

    def test_escapes_quotes(self):
        assert EPUBGenerator._xml_escape('"hi"') == "&quot;hi&quot;"

    def test_escapes_angle_brackets(self):
        assert EPUBGenerator._xml_escape("<tag>") == "&lt;tag&gt;"

    def test_combined(self):
        result = EPUBGenerator._xml_escape('A & B <c> "d"')
        assert result == "A &amp; B &lt;c&gt; &quot;d&quot;"


class TestMarkdownToXhtml:
    def test_basic_markdown(self):
        g = EPUBGenerator()
        html = g.markdown_to_xhtml("**bold**")
        assert "<strong>bold</strong>" in html

    def test_escapes_loose_ampersands(self):
        g = EPUBGenerator()
        # Markdown won't escape stray & — generator must
        html = g.markdown_to_xhtml("Tom & Jerry")
        assert "Tom &amp; Jerry" in html

    def test_keeps_existing_entities(self):
        g = EPUBGenerator()
        html = g.markdown_to_xhtml("Tom &amp; Jerry")
        # Shouldn't double-escape
        assert "&amp;amp;" not in html

    def test_strips_non_bmp(self):
        g = EPUBGenerator()
        # 🚀 is in supplementary plane (>0xFFFF)
        html = g.markdown_to_xhtml("Rocket 🚀 launch")
        assert "🚀" not in html
        assert "launch" in html

    def test_right_alignment_marker(self):
        g = EPUBGenerator()
        html = g.markdown_to_xhtml("Left [RIGHT]right[/RIGHT]")
        assert 'float: right' in html


class TestGenerateEpub:
    def test_creates_valid_zip(self, tmp_path):
        out = tmp_path / "x.epub"
        g = EPUBGenerator(title="Test", author="Author")
        g.generate_epub("# Title\n\nContent", str(out))
        assert out.exists()
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            assert "mimetype" in names
            assert "META-INF/container.xml" in names
            assert "OEBPS/content.opf" in names
            assert "OEBPS/main.xhtml" in names
            assert "OEBPS/toc.ncx" in names
            assert "OEBPS/style.css" in names

    def test_mimetype_uncompressed(self, tmp_path):
        out = tmp_path / "x.epub"
        EPUBGenerator().generate_epub("# T", str(out))
        with zipfile.ZipFile(out) as z:
            info = z.getinfo("mimetype")
            assert info.compress_type == zipfile.ZIP_STORED
            assert z.read("mimetype").decode() == "application/epub+zip"

    def test_title_escaped_in_opf(self, tmp_path):
        out = tmp_path / "x.epub"
        g = EPUBGenerator(title="A & B")
        g.generate_epub("# T", str(out))
        with zipfile.ZipFile(out) as z:
            opf = z.read("OEBPS/content.opf").decode()
            assert "A &amp; B" in opf
            assert "A & B" not in opf  # raw ampersand not present

    def test_author_in_opf(self, tmp_path):
        out = tmp_path / "x.epub"
        EPUBGenerator(author="John Doe").generate_epub("# T", str(out))
        with zipfile.ZipFile(out) as z:
            opf = z.read("OEBPS/content.opf").decode()
            assert "John Doe" in opf

    def test_uuid_consistent_across_files(self, tmp_path):
        out = tmp_path / "x.epub"
        g = EPUBGenerator()
        g.generate_epub("# T", str(out))
        with zipfile.ZipFile(out) as z:
            ncx = z.read("OEBPS/toc.ncx").decode()
            opf = z.read("OEBPS/content.opf").decode()
            assert g.uuid in ncx
            assert g.uuid in opf

    def test_main_xhtml_contains_content(self, tmp_path):
        out = tmp_path / "x.epub"
        EPUBGenerator().generate_epub("# Hello\n\n**Body**", str(out))
        with zipfile.ZipFile(out) as z:
            xhtml = z.read("OEBPS/main.xhtml").decode()
            assert "<strong>Body</strong>" in xhtml


class TestEpubMain:
    def test_main_generates(self, tmp_path, monkeypatch):
        md = tmp_path / "in.md"
        md.write_text("# T\n\nBody")
        out = tmp_path / "out.epub"
        monkeypatch.setattr(
            "sys.argv", ["epub.py", "--input", str(md), "--output", str(out)]
        )
        assert main() == 0
        assert out.exists()

    def test_main_custom_title(self, tmp_path, monkeypatch):
        md = tmp_path / "in.md"
        md.write_text("# T")
        out = tmp_path / "out.epub"
        monkeypatch.setattr(
            "sys.argv",
            ["epub.py", "--input", str(md), "--output", str(out),
             "--title", "Custom Title"],
        )
        assert main() == 0
        with zipfile.ZipFile(out) as z:
            opf = z.read("OEBPS/content.opf").decode()
            assert "Custom Title" in opf

    def test_main_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["epub.py", "--input", str(tmp_path / "missing"),
             "--output", str(tmp_path / "out.epub")],
        )
        assert main() == 1
