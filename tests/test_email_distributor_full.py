# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Comprehensive tests for email_distributor module."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.email_distributor import EmailDistributor


@pytest.fixture
def distributor():
    return EmailDistributor(sender_email="sender@x.com", sender_password="pw")


class TestMarkdownToHtml:
    def test_converts_basic_markdown(self, distributor):
        html = distributor._markdown_to_html("# Title\n\nSome **bold** text.")
        assert "<h1>" in html
        assert "<strong>bold</strong>" in html

    def test_sanitizes_unsafe_tags(self, distributor):
        # nh3 should strip script tags
        md = "Hello <script>alert('xss')</script> world"
        html = distributor._markdown_to_html(md)
        assert "<script>" not in html

    def test_renders_tables(self, distributor):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        html = distributor._markdown_to_html(md)
        assert "<table>" in html

    def test_right_alignment_marker(self, distributor):
        md = "Left text [RIGHT]aligned right[/RIGHT]"
        html = distributor._markdown_to_html(md)
        assert 'float: right' in html
        assert "aligned right" in html

    def test_includes_styling_template(self, distributor):
        html = distributor._markdown_to_html("# T\n")
        assert "<!DOCTYPE html>" in html
        assert ".container" in html

    def test_includes_footer(self, distributor):
        html = distributor._markdown_to_html("Hi")
        assert "Atlas Morning Briefing" in html
        assert "footer" in html


class TestSendKindle:
    def test_missing_file_returns_false(self, distributor, tmp_path):
        assert distributor.send_kindle("kindle@x.com", str(tmp_path / "nope.pdf")) is False

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_send_pdf(self, mock_smtp_cls, distributor, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server

        ok = distributor.send_kindle("kindle@x.com", str(pdf_path), subject="Test")
        assert ok is True
        server.send_message.assert_called_once()

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_epub_uses_convert_subject(self, mock_smtp_cls, distributor, tmp_path):
        epub_path = tmp_path / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04 fake epub")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server

        ok = distributor.send_kindle("kindle@x.com", str(epub_path), subject="Test")
        assert ok is True
        # Subject should be overridden to "Convert" for EPUBs
        msg = server.send_message.call_args.args[0]
        assert msg["Subject"] == "Convert"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_default_subject_from_filename(self, mock_smtp_cls, distributor, tmp_path):
        pdf_path = tmp_path / "MyBriefing.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        distributor.send_kindle("kindle@x.com", str(pdf_path))
        msg = server.send_message.call_args.args[0]
        assert msg["Subject"] == "MyBriefing"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_unknown_extension_uses_octet_stream(self, mock_smtp_cls, distributor, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"binary")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        assert distributor.send_kindle("k@x.com", str(f)) is True

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_smtp_failure_returns_false(self, mock_smtp_cls, distributor, tmp_path):
        pdf_path = tmp_path / "x.pdf"
        pdf_path.write_bytes(b"%PDF")
        mock_smtp_cls.side_effect = OSError("SMTP unreachable")
        assert distributor.send_kindle("k@x.com", str(pdf_path)) is False

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_subject_strips_newlines(self, mock_smtp_cls, distributor, tmp_path):
        """Multi-line subjects can break SMTP — must be flattened."""
        pdf_path = tmp_path / "x.pdf"
        pdf_path.write_bytes(b"%PDF")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        distributor.send_kindle("k@x.com", str(pdf_path), subject="Line1\nLine2\rLine3")
        msg = server.send_message.call_args.args[0]
        assert "\n" not in msg["Subject"]
        assert "\r" not in msg["Subject"]


class TestSendHtmlEmail:
    def test_empty_recipients(self, distributor):
        assert distributor.send_html_email([], "md") == {}

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_sends_html(self, mock_smtp_cls, distributor):
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server

        results = distributor.send_html_email(
            ["a@x.com", "b@y.com"], "# Hello\nbody"
        )
        assert results == {"a@x.com": True, "b@y.com": True}
        # Single send_message call with both recipients in To
        assert server.send_message.call_count == 1

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_includes_pdf_attachment(self, mock_smtp_cls, distributor, tmp_path):
        pdf_path = tmp_path / "att.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server

        distributor.send_html_email(
            ["a@x.com"], "body", attachment_path=str(pdf_path)
        )
        # Confirm attachment was included
        msg = server.send_message.call_args.args[0]
        # mixed multipart when attachment present
        assert msg.get_content_type() == "multipart/mixed"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_includes_epub_attachment(self, mock_smtp_cls, distributor, tmp_path):
        f = tmp_path / "att.epub"
        f.write_bytes(b"PK\x03\x04")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        distributor.send_html_email(["a@x.com"], "body", attachment_path=str(f))
        msg = server.send_message.call_args.args[0]
        assert msg.get_content_type() == "multipart/mixed"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_missing_attachment_skipped(self, mock_smtp_cls, distributor, tmp_path):
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        # File doesn't exist; should not raise, just skip attachment
        distributor.send_html_email(
            ["a@x.com"], "body", attachment_path=str(tmp_path / "nope.pdf")
        )
        # Still sends as alternative (no mixed)
        msg = server.send_message.call_args.args[0]
        assert msg.get_content_type() == "multipart/alternative"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_smtp_failure_returns_false_for_all(self, mock_smtp_cls, distributor):
        mock_smtp_cls.side_effect = OSError("SMTP down")
        results = distributor.send_html_email(["a@x.com", "b@x.com"], "body")
        assert results == {"a@x.com": False, "b@x.com": False}

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_default_subject(self, mock_smtp_cls, distributor):
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        distributor.send_html_email(["a@x.com"], "body")
        msg = server.send_message.call_args.args[0]
        assert msg["Subject"] == "Atlas Morning Briefing"


class TestDistribute:
    def test_dry_run_skips_everything(self, distributor):
        results = distributor.distribute(
            config={"kindle_email": "k@x.com", "email_recipients": ["a@x.com"]},
            markdown_content="md",
            pdf_path="/some.pdf",
            dry_run=True,
        )
        assert results == {}

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_prefers_epub_for_kindle(self, mock_smtp_cls, distributor, tmp_path):
        epub = tmp_path / "x.epub"
        pdf = tmp_path / "x.pdf"
        epub.write_bytes(b"PK\x03\x04")
        pdf.write_bytes(b"%PDF")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server

        results = distributor.distribute(
            config={"kindle_email": "k@x.com"},
            markdown_content="md",
            pdf_path=str(pdf),
            epub_path=str(epub),
        )
        # Verify EPUB was used (Subject would be "Convert")
        kindle_msg = None
        for call in server.send_message.call_args_list:
            m = call.args[0]
            if m["To"] == "k@x.com":
                kindle_msg = m
        assert kindle_msg is not None
        assert kindle_msg["Subject"] == "Convert"

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_falls_back_to_pdf_if_no_epub(self, mock_smtp_cls, distributor, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF")
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        results = distributor.distribute(
            config={"kindle_email": "k@x.com"},
            markdown_content="md",
            pdf_path=str(pdf),
        )
        assert "kindle:k@x.com" in results

    @patch("scripts.email_distributor.smtplib.SMTP")
    def test_handles_comma_separated_recipients(self, mock_smtp_cls, distributor):
        server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = server
        results = distributor.distribute(
            config={"email_recipients": ["a@x.com,b@x.com", "c@x.com"]},
            markdown_content="md",
        )
        assert len(results) == 3
        for k in ["a@x.com", "b@x.com", "c@x.com"]:
            assert k in results
