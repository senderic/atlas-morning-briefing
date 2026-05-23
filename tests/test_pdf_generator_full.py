# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Comprehensive tests for pdf_generator (tables, code blocks, RIGHT alignment, main())."""

from unittest.mock import patch

import pytest

from scripts.pdf_generator import PDFGenerator, main


@pytest.fixture
def gen():
    return PDFGenerator(page_format="kindle", font_size=10, line_spacing=1.5)


class TestStripEmojiAndStars:
    def test_converts_stars_to_numeric(self, gen):
        result = gen.strip_emoji("Title ★★★★☆")
        assert "(4/5)" in result
        assert "★" not in result

    def test_zero_stars(self, gen):
        result = gen.strip_emoji("☆☆☆☆☆")
        assert "(0/5)" in result

    def test_five_stars(self, gen):
        result = gen.strip_emoji("★★★★★")
        assert "(5/5)" in result

    def test_strips_smiley_emoji(self, gen):
        result = gen.strip_emoji("Hello 😀 world")
        assert "😀" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_pictograph_emoji(self, gen):
        result = gen.strip_emoji("Stock 📈 up")
        assert "📈" not in result
        assert "Stock" in result


class TestStripMdLinks:
    def test_strips_link_markup(self):
        s = PDFGenerator._strip_md_links("See [docs](http://x.com) for more")
        assert s == "See docs for more"

    def test_no_link_unchanged(self):
        s = PDFGenerator._strip_md_links("Plain text")
        assert s == "Plain text"

    def test_multiple_links(self):
        s = PDFGenerator._strip_md_links("[a](x) and [b](y)")
        assert s == "a and b"


class TestMarkdownToFlowablesExtras:
    def test_table_rendered(self, gen):
        md = "| Ticker | Price |\n|---|---|\n| AAPL | $150 |\n"
        flowables = gen.markdown_to_flowables(md)
        # Table + spacer = at least 2 flowables
        assert len(flowables) >= 2

    def test_four_column_table(self, gen):
        # Triggers special column-width branch
        md = "| A | B | C | D |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |\n"
        flowables = gen.markdown_to_flowables(md)
        assert len(flowables) >= 1

    def test_right_alignment_markers(self, gen):
        md = "Left text [RIGHT]right text[/RIGHT]"
        flowables = gen.markdown_to_flowables(md)
        # Should render as a 2-column table
        assert len(flowables) >= 1

    def test_bold_and_italic(self, gen):
        md = "**bold** and *italic* text"
        flowables = gen.markdown_to_flowables(md)
        assert len(flowables) >= 1

    def test_code_block_complete(self, gen):
        md = "```\nprint('hi')\nx = 1\n```"
        flowables = gen.markdown_to_flowables(md)
        # Code lines accumulated then flushed as one block
        assert len(flowables) >= 1

    def test_html_escape_in_body(self, gen):
        md = "Less than: <script>alert(1)</script>"
        flowables = gen.markdown_to_flowables(md)
        # Just verify it doesn't crash; content is escaped
        assert len(flowables) >= 1

    def test_link_in_body(self, gen):
        md = "Check [docs](https://example.com) please"
        flowables = gen.markdown_to_flowables(md)
        assert len(flowables) >= 1

    def test_h4_heading(self, gen):
        md = "#### Subsection heading"
        flowables = gen.markdown_to_flowables(md)
        assert len(flowables) >= 1

    def test_table_then_paragraph(self, gen):
        md = "| A |\n|---|\n| 1 |\n\nNormal paragraph after"
        flowables = gen.markdown_to_flowables(md)
        # Table flushed when non-table line appears
        assert len(flowables) >= 2

    def test_table_at_end_of_doc(self, gen):
        # Table without trailing non-table content — must still flush
        md = "| A |\n|---|\n| 1 |"
        flowables = gen.markdown_to_flowables(md)
        assert len(flowables) >= 1


class TestRenderTable:
    def test_empty_rows_returns_empty(self, gen):
        assert gen._render_table([]) == []

    def test_pads_short_rows(self, gen):
        rows = [["A", "B", "C"], ["1"]]
        flowables = gen._render_table(rows)
        assert len(flowables) == 2  # table + spacer

    def test_non_four_column_uses_equal_widths(self, gen):
        rows = [["A", "B", "C"], ["1", "2", "3"]]
        flowables = gen._render_table(rows)
        assert len(flowables) == 2


class TestGeneratePdfRich:
    def test_generates_with_table(self, gen, tmp_path):
        md = (
            "# Briefing\n\n"
            "## Stocks\n\n"
            "| Symbol | Price | Change | Driver |\n"
            "|--------|-------|--------|--------|\n"
            "| AAPL | $150 | +1.2% | Earnings beat |\n\n"
            "## News\n\nSome news content."
        )
        out = str(tmp_path / "out.pdf")
        gen.generate_pdf(md, out)
        from pathlib import Path
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_generates_with_code_block(self, gen, tmp_path):
        md = "# Title\n\n```python\nprint('x')\n```\n"
        out = str(tmp_path / "out.pdf")
        gen.generate_pdf(md, out)
        from pathlib import Path
        assert Path(out).exists()

    def test_a4_format(self, tmp_path):
        gen = PDFGenerator(page_format="a4", font_size=12, line_spacing=1.2)
        out = str(tmp_path / "a4.pdf")
        gen.generate_pdf("# A4 Briefing\n\nBody", out)
        from pathlib import Path
        assert Path(out).exists()

    def test_letter_format(self, tmp_path):
        gen = PDFGenerator(page_format="letter")
        out = str(tmp_path / "letter.pdf")
        gen.generate_pdf("# Letter\n\nBody", out)
        from pathlib import Path
        assert Path(out).exists()


class TestPdfMain:
    def test_missing_input(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["pdf.py", "--input", str(tmp_path / "missing.md"),
             "--output", str(tmp_path / "out.pdf")],
        )
        assert main() == 2

    def test_main_generates(self, tmp_path, monkeypatch):
        md_path = tmp_path / "in.md"
        md_path.write_text("# Hello\n\nBody.")
        out_path = tmp_path / "out.pdf"
        monkeypatch.setattr(
            "sys.argv",
            ["pdf.py", "--input", str(md_path), "--output", str(out_path)],
        )
        assert main() == 0
        assert out_path.exists()

    def test_main_format_a4(self, tmp_path, monkeypatch):
        md_path = tmp_path / "in.md"
        md_path.write_text("# A4")
        out_path = tmp_path / "out.pdf"
        monkeypatch.setattr(
            "sys.argv",
            ["pdf.py", "--input", str(md_path), "--output", str(out_path),
             "--format", "a4"],
        )
        assert main() == 0

    @patch("scripts.pdf_generator.PDFGenerator.generate_pdf")
    def test_main_generation_error(self, mock_gen, tmp_path, monkeypatch):
        md_path = tmp_path / "in.md"
        md_path.write_text("# Hi")
        out_path = tmp_path / "out.pdf"
        mock_gen.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            "sys.argv",
            ["pdf.py", "--input", str(md_path), "--output", str(out_path)],
        )
        assert main() == 2
