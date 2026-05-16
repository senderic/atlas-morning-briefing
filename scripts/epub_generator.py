#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
EPUB generator.

Generates Kindle-optimized EPUBs from markdown content.
Uses a strictly compliant EPUB 2.0 structure for maximum "Send to Kindle" compatibility.
"""

import argparse
import logging
import os
import sys
import zipfile
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown

logger = logging.getLogger(__name__)


class EPUBGenerator:
    """Generates Kindle-optimized EPUBs from markdown."""

    def __init__(self, title: str = "Morning Briefing", author: str = "Atlas"):
        """
        Initialize EPUBGenerator.

        Args:
            title: Book title
            author: Book author
        """
        self.raw_title = title
        self.title = self._xml_escape(title)
        self.author = self._xml_escape(author)
        self.uuid = str(uuid.uuid4())
        # Date format for dc:date
        self.date_str = datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape special characters for XML."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")

    def markdown_to_xhtml(self, markdown_content: str) -> str:
        """
        Convert markdown to strictly valid XHTML 1.1.

        Args:
            markdown_content: Markdown text

        Returns:
            XHTML body string
        """
        # Convert markdown to XHTML
        # output_format="xhtml" ensures self-closing tags like <br />
        html_content = markdown.markdown(
            markdown_content,
            extensions=["tables", "fenced_code", "nl2br"],
            output_format="xhtml",
        )

        # Handle [RIGHT] markers for alignment
        html_content = html_content.replace("[RIGHT]", '<span style="float: right;">').replace("[/RIGHT]", "</span>")
        
        # Aggressive cleaning for Kindle parser
        # 1. Ensure all & are escaped to &amp;
        html_content = re.sub(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[a-fA-F\d]+);)', '&amp;', html_content)
        
        # 2. Remove any emoji or characters outside Basic Multilingual Plane
        # (Kindle conversion sometimes chokes on 4-byte UTF-8). Log the count
        # so silent loss of emoji / non-BMP CJK is at least visible.
        dropped = sum(1 for c in html_content if ord(c) > 0xFFFF)
        if dropped:
            logger.warning(f"Dropped {dropped} non-BMP characters from EPUB content")
            html_content = "".join(c for c in html_content if ord(c) <= 0xFFFF)

        return html_content

    def generate_epub(self, markdown_content: str, output_path: str) -> None:
        """
        Generate EPUB from markdown content.

        Args:
            markdown_content: Markdown text
            output_path: Output EPUB file path
        """
        logger.info(f"Generating Kindle-optimized EPUB: {output_path}")
        
        xhtml_content = self.markdown_to_xhtml(markdown_content)
        
        with zipfile.ZipFile(output_path, 'w') as epub:
            # 1. mimetype (MUST be first and uncompressed)
            epub.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            
            # 2. META-INF/container.xml
            container = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
            epub.writestr("META-INF/container.xml", container)
            
            # 3. OEBPS/style.css
            css = """
body { font-family: sans-serif; line-height: 1.5; margin: 5%; }
h1 { text-align: center; color: #1a1a1a; }
h2 { color: #1f6feb; border-bottom: 1px solid #e1e4e8; padding-bottom: 5px; margin-top: 1.5em; }
h3 { color: #24292f; margin-top: 1.2em; }
h4 { color: #57606a; font-size: 0.85em; margin-top: 1em; text-transform: uppercase; letter-spacing: 0.05em; }
h4 + p { font-size: 0.9em; color: #57606a; margin-top: 0; }
p { margin: 0.5em 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { border: 1px solid #d0d7de; padding: 6px; text-align: left; }
th { background-color: #f0f3f6; }
pre { background-color: #f6f8fa; padding: 10px; border-radius: 5px; overflow-x: auto; font-family: monospace; }
code { font-family: monospace; background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; }
"""
            epub.writestr("OEBPS/style.css", css)
            
            # 4. OEBPS/main.xhtml
            main_xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{self.title}</title>
    <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
    {xhtml_content}
</body>
</html>'''
            epub.writestr("OEBPS/main.xhtml", main_xhtml)
            
            # 5. OEBPS/toc.ncx (Table of Contents)
            ncx = f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{self.uuid}"/>
        <meta name="dtb:depth" content="1"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle><text>{self.title}</text></docTitle>
    <navMap>
        <navPoint id="navpoint-1" playOrder="1">
            <navLabel><text>Morning Briefing</text></navLabel>
            <content src="main.xhtml"/>
        </navPoint>
    </navMap>
</ncx>'''
            epub.writestr("OEBPS/toc.ncx", ncx)
            
            # 6. OEBPS/content.opf (Package File)
            opf = f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="2.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
        <dc:title>{self.title}</dc:title>
        <dc:creator opf:role="aut">{self.author}</dc:creator>
        <dc:identifier id="bookid">{self.uuid}</dc:identifier>
        <dc:language>en</dc:language>
        <dc:date>{self.date_str}</dc:date>
    </metadata>
    <manifest>
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
        <item id="style" href="style.css" media-type="text/css"/>
        <item id="main" href="main.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="main"/>
    </spine>
</package>'''
            epub.writestr("OEBPS/content.opf", opf)

        logger.info(f"EPUB generation complete: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Kindle-optimized EPUB")
    parser.add_argument("--input", required=True, help="Input markdown path")
    parser.add_argument("--output", required=True, help="Output EPUB path")
    parser.add_argument("--title", default="Morning Briefing", help="Book title")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            md_content = f.read()
        generator = EPUBGenerator(title=args.title)
        generator.generate_epub(md_content, args.output)
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
