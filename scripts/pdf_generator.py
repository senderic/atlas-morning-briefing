#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
PDF generator.

Generates Kindle-optimized PDFs from markdown content.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch as inch_unit
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generates Kindle-optimized PDFs from markdown."""

    # Page sizes (width, height) in inches
    PAGE_SIZES = {
        "kindle": (6 * inch_unit, 8 * inch_unit),
        "a4": (8.27 * inch_unit, 11.69 * inch_unit),
        "letter": (8.5 * inch_unit, 11 * inch_unit),
    }

    def __init__(
        self,
        page_format: str = "kindle",
        font_size: int = 10,
        line_spacing: float = 1.5,
        include_toc: bool = True,
    ):
        """
        Initialize PDFGenerator.

        Args:
            page_format: Page format (kindle/a4/letter)
            font_size: Base font size
            line_spacing: Line spacing multiplier
            include_toc: Whether to include table of contents
        """
        self.page_format = page_format
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.include_toc = include_toc

        # Get page size
        self.page_size = self.PAGE_SIZES.get(page_format, self.PAGE_SIZES["kindle"])

        # Initialize styles
        self.styles = self._create_styles()

    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """
        Create paragraph styles.

        Returns:
            Dictionary of styles
        """
        base_styles = getSampleStyleSheet()

        styles = {
            "title": ParagraphStyle(
                "CustomTitle",
                parent=base_styles["Title"],
                fontSize=self.font_size + 8,
                spaceAfter=10,
                alignment=1,  # Center
            ),
            "heading1": ParagraphStyle(
                "CustomHeading1",
                parent=base_styles["Heading1"],
                fontSize=self.font_size + 4,
                spaceAfter=4,
                spaceBefore=8,
                textColor=colors.HexColor("#1a1a1a"),
            ),
            "heading2": ParagraphStyle(
                "CustomHeading2",
                parent=base_styles["Heading2"],
                fontSize=self.font_size + 2,
                spaceAfter=3,
                spaceBefore=6,
                textColor=colors.HexColor("#2a2a2a"),
            ),
            "heading3": ParagraphStyle(
                "CustomHeading3",
                parent=base_styles["Heading3"],
                fontSize=self.font_size + 1,
                spaceAfter=2,
                spaceBefore=4,
                textColor=colors.HexColor("#3a3a3a"),
            ),
            "heading4": ParagraphStyle(
                "CustomHeading4",
                parent=base_styles["Heading4"],
                fontSize=self.font_size - 1,
                spaceAfter=1,
                spaceBefore=4,
                textColor=colors.HexColor("#57606a"),
                textTransform="uppercase",
            ),
            "body": ParagraphStyle(
                "CustomBody",
                parent=base_styles["BodyText"],
                fontSize=self.font_size,
                leading=self.font_size * self.line_spacing,
                spaceAfter=3,
            ),
            "code": ParagraphStyle(
                "CustomCode",
                parent=base_styles["Code"],
                fontSize=self.font_size - 1,
                fontName="Courier",
                leftIndent=20,
                rightIndent=20,
                spaceAfter=6,
                spaceBefore=6,
                backColor=colors.HexColor("#f5f5f5"),
            ),
        }

        return styles

    def strip_emoji(self, text: str) -> str:
        """
        Strip emoji characters and convert star ratings for PDF compatibility.

        Args:
            text: Input text

        Returns:
            Text with emoji removed and stars converted
        """
        # Convert star ratings to numeric: ★★★★☆ → (4/5)
        star_pattern = re.compile(r"[★☆]{5}")
        match = star_pattern.search(text)
        if match:
            stars = match.group()
            filled = stars.count("★")
            text = text.replace(stars, f"({filled}/5)")

        # Remove emoji and other special unicode characters
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    @staticmethod
    def _strip_md_links(text: str) -> str:
        """Strip markdown links [text](url) to just text."""
        return re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    def parse_markdown_line(self, line: str) -> Tuple[str, str]:
        """
        Parse a markdown line and return (type, content).

        Args:
            line: Markdown line

        Returns:
            Tuple of (line_type, content)
        """
        line = line.rstrip()

        if not line:
            return ("empty", "")

        # Headers
        if line.startswith("# "):
            return ("h1", self._strip_md_links(line[2:].strip()))
        elif line.startswith("## "):
            return ("h2", self._strip_md_links(line[3:].strip()))
        elif line.startswith("### "):
            return ("h3", self._strip_md_links(line[4:].strip()))
        elif line.startswith("#### "):
            return ("h4", self._strip_md_links(line[5:].strip()))

        # Code blocks (simplified)
        if line.startswith("```"):
            return ("code_fence", line[3:].strip())

        # Lists (simplified - treat as body text)
        if line.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", line):
            return ("body", line)

        # Default to body text
        return ("body", line)

    def markdown_to_flowables(self, markdown_content: str) -> List[Any]:
        """
        Convert markdown content to ReportLab flowables.

        Args:
            markdown_content: Markdown text

        Returns:
            List of flowables
        """
        flowables = []
        lines = markdown_content.split("\n")

        in_code_block = False
        code_lines = []
        in_table = False
        table_rows = []

        for line in lines:
            # Strip emoji
            line = self.strip_emoji(line)

            # Table detection
            if "|" in line and line.strip().startswith("|"):
                stripped = line.strip()
                # Skip separator rows like |---|---|
                if all(c in "|-: " for c in stripped):
                    continue
                # Parse table row
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if cells:
                    table_rows.append(cells)
                    in_table = True
                    continue
            elif in_table:
                # End of table — render it
                flowables.extend(self._render_table(table_rows))
                table_rows = []
                in_table = False

            line_type, content = self.parse_markdown_line(line)

            if line_type == "code_fence":
                if in_code_block:
                    # End code block
                    if code_lines:
                        code_text = "\n".join(code_lines)
                        flowables.append(
                            Paragraph(
                                code_text.replace("\n", "<br/>"),
                                self.styles["code"],
                            )
                        )
                        code_lines = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(content)
                continue

            if line_type == "h1":
                flowables.append(Paragraph(content, self.styles["heading1"]))
            elif line_type == "h2":
                flowables.append(Paragraph(content, self.styles["heading2"]))
            elif line_type == "h3":
                flowables.append(Paragraph(content, self.styles["heading3"]))
            elif line_type == "h4":
                flowables.append(Paragraph(content, self.styles["heading4"]))
            elif line_type == "body":
                if content:
                    # Special handling for [RIGHT] alignment markers (usually for subtitles)
                    if "[RIGHT]" in content and "[/RIGHT]" in content:
                        parts = content.split("[RIGHT]")
                        left_text = parts[0].strip()
                        right_text = parts[1].split("[/RIGHT]")[0].strip()
                        
                        # Process both parts as markdown (e.g. bold/italic)
                        # We do it manually here because this is a special case
                        def process_md(text):
                            text = re.sub(r"\*\*(.+?)\*\*", "<b>\\1</b>", text)
                            text = text.replace("**", "")
                            text = re.sub(r"\*(.+?)\*", "<i>\\1</i>", text)
                            return text

                        left_text = process_md(left_text)
                        right_text = process_md(right_text)
                        
                        # Create a two-column table with no borders for alignment
                        page_width = self.page_size[0] - 1.0 * inch_unit
                        data = [[
                            Paragraph(left_text, self.styles["body"]),
                            Paragraph(right_text, ParagraphStyle("RightAlign", parent=self.styles["body"], alignment=2)) # 2 = Right
                        ]]
                        table = Table(data, colWidths=[page_width * 0.6, page_width * 0.4])
                        table.setStyle(TableStyle([
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]))
                        flowables.append(table)
                        continue

                    # Convert markdown formatting to placeholders before
                    # HTML-escaping, so user content is escaped but our
                    # markup tags survive.  \x00 = '<'  \x01 = '>'
                    content = re.sub(
                        r"\*\*(.+?)\*\*", "\x00b\x01\\1\x00/b\x01", content
                    )
                    # Strip any remaining unmatched ** (e.g., stray bold markers)
                    content = content.replace("**", "")
                    content = re.sub(
                        r"\*(.+?)\*", "\x00i\x01\\1\x00/i\x01", content
                    )
                    content = re.sub(
                        r"\[([^\]]+)\]\(([^\)]+)\)",
                        '\x00a href="\\2" color="blue"\x01\\1\x00/a\x01',
                        content,
                    )
                    # Escape HTML special characters in user content
                    content = (
                        content.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    # Restore our markup placeholders to real tags
                    content = content.replace("\x00", "<").replace("\x01", ">")
                    flowables.append(Paragraph(content, self.styles["body"]))
            elif line_type == "empty":
                flowables.append(Spacer(1, 0.05 * inch_unit))

        # Flush remaining table
        if in_table and table_rows:
            flowables.extend(self._render_table(table_rows))

        return flowables

    def _render_table(self, rows: List[List[str]]) -> List[Any]:
        """
        Render a markdown table as a ReportLab Table with word wrapping.

        Args:
            rows: List of rows, each row is a list of cell strings.

        Returns:
            List of flowables (table + spacer).
        """
        if not rows:
            return []

        # Calculate column widths — give more space to last column (Driver)
        page_width = self.page_size[0] - 1.0 * inch_unit  # margins
        num_cols = max(len(r) for r in rows)

        if num_cols == 4:
            # Ticker | Price | Change | Driver
            col_widths = [
                page_width * 0.18,
                page_width * 0.20,
                page_width * 0.18,
                page_width * 0.44,
            ]
        else:
            col_widths = [page_width / num_cols] * num_cols

        # Build cell styles
        cell_style = ParagraphStyle(
            "TableCell",
            fontSize=self.font_size - 1,
            fontName="Helvetica",
            leading=self.font_size + 1,
        )
        header_style = ParagraphStyle(
            "TableHeader",
            fontSize=self.font_size - 1,
            fontName="Helvetica-Bold",
            leading=self.font_size + 1,
        )

        # Convert cells to Paragraphs for word wrapping
        table_data = []
        for i, row in enumerate(rows):
            style = header_style if i == 0 else cell_style
            para_row = []
            for cell in row:
                cell = cell.replace("**", "")
                para_row.append(Paragraph(cell, style))
            # Pad
            while len(para_row) < num_cols:
                para_row.append(Paragraph("", cell_style))
            table_data.append(para_row)

        table = Table(table_data, colWidths=col_widths)

        style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]

        table.setStyle(TableStyle(style_commands))

        return [table, Spacer(1, 0.05 * inch_unit)]

    def generate_pdf(self, markdown_content: str, output_path: str) -> None:
        """
        Generate PDF from markdown content.

        Args:
            markdown_content: Markdown text
            output_path: Output PDF file path
        """
        logger.info(f"Generating PDF: {output_path}")

        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.5 * inch_unit,
            leftMargin=0.5 * inch_unit,
            topMargin=0.75 * inch_unit,
            bottomMargin=0.75 * inch_unit,
        )

        # Convert markdown to flowables
        flowables = self.markdown_to_flowables(markdown_content)

        # Build PDF
        doc.build(flowables)
        logger.info(f"PDF generated successfully: {output_path}")


def main() -> int:
    """
    Main entry point for pdf_generator.

    Returns:
        Exit code (0 for success, 2 for failure)
    """
    parser = argparse.ArgumentParser(description="Generate PDF from markdown")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input markdown file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PDF file path",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="kindle",
        choices=["kindle", "a4", "letter"],
        help="Page format",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=10,
        help="Base font size",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.5,
        help="Line spacing multiplier",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Read input markdown
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 2

        with open(input_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

    except IOError as e:
        logger.error(f"Failed to read input file: {e}")
        return 2

    # Generate PDF
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = PDFGenerator(
            page_format=args.format,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
        )
        generator.generate_pdf(markdown_content, str(output_path))

        return 0

    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
