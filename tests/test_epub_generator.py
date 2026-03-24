import os
import unittest
from pathlib import Path
import zipfile
from scripts.epub_generator import EPUBGenerator

class TestEPUBGenerator(unittest.TestCase):
    def setUp(self):
        self.output_path = "test_output.epub"
        self.generator = EPUBGenerator(title="Test Title", author="Test Author")

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_generate_epub_creates_file(self):
        markdown_content = "# Hello World\nThis is a test."
        self.generator.generate_epub(markdown_content, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertTrue(os.path.getsize(self.output_path) > 0)

    def test_epub_structure(self):
        markdown_content = "# Test\nContent"
        self.generator.generate_epub(markdown_content, self.output_path)
        
        with zipfile.ZipFile(self.output_path, 'r') as epub:
            file_list = epub.namelist()
            self.assertIn('mimetype', file_list)
            self.assertIn('META-INF/container.xml', file_list)
            self.assertIn('OEBPS/content.opf', file_list)
            self.assertIn('OEBPS/content.xhtml', file_list)
            self.assertIn('OEBPS/toc.ncx', file_list)
            self.assertIn('OEBPS/style.css', file_list)
            
            # Check mimetype content
            with epub.open('mimetype') as f:
                self.assertEqual(f.read().decode('utf-8'), 'application/epub+zip')

    def test_markdown_to_html(self):
        markdown_content = "**Bold** and *Italic*"
        html = self.generator.markdown_to_html(markdown_content)
        self.assertIn("<strong>Bold</strong>", html)
        self.assertIn("<em>Italic</em>", html)

if __name__ == "__main__":
    unittest.main()
