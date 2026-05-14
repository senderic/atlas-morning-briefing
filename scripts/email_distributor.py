#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Email distribution module.

Sends briefings via multiple channels:
- Kindle: PDF/EPUB attachment
- Email list: Rich HTML format for regular email clients
"""

import logging
import re
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import markdown

try:
    import nh3
    HAS_NH3 = True
except ImportError:
    HAS_NH3 = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class EmailDistributor:
    """Distributes briefings via email to multiple recipients."""

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    def __init__(
        self,
        sender_email: str,
        sender_password: str,
    ):
        """
        Initialize EmailDistributor.

        Args:
            sender_email: Gmail address.
            sender_password: Gmail app password.
        """
        self.sender_email = sender_email
        self.sender_password = sender_password

    def _connect_smtp(self) -> smtplib.SMTP:
        """Create and authenticate SMTP connection."""
        server = smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT)
        server.starttls()
        server.login(self.sender_email, self.sender_password)
        return server

    def _markdown_to_html(self, md_content: str) -> str:
        """
        Convert markdown briefing to rich HTML email.

        Args:
            md_content: Markdown string.

        Returns:
            Complete HTML document string.
        """
        # Convert markdown to HTML body
        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "nl2br"],
        )

        # Sanitize HTML to prevent XSS from untrusted content
        if HAS_NH3:
            html_body = nh3.clean(html_body)
        else:
            logger.warning("nh3 not installed; HTML email output is not sanitized")

        # Wrap in a styled HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 680px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f8f9fa;
  }}
  .container {{
    background-color: #ffffff;
    border-radius: 8px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  h1 {{
    color: #0d1117;
    font-size: 24px;
    border-bottom: 2px solid #58a6ff;
    padding-bottom: 8px;
    margin-top: 0;
  }}
  h2 {{
    color: #1f6feb;
    font-size: 18px;
    margin-top: 28px;
    border-bottom: 1px solid #e1e4e8;
    padding-bottom: 6px;
  }}
  h3 {{
    color: #24292f;
    font-size: 15px;
    margin-top: 20px;
    margin-bottom: 4px;
  }}
  p {{ margin: 8px 0; font-size: 14px; }}
  a {{ color: #1f6feb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e1e4e8;
    font-size: 12px;
    color: #8b949e;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
{html_body}
<div class="footer">
  Atlas Morning Briefing<br>
  <a href="https://github.com/senderic/atlas-morning-briefing">GitHub</a>
</div>
</div>
</body>
</html>"""
        return html

    def send_kindle(
        self,
        kindle_email: str,
        file_path: str,
        subject: Optional[str] = None,
    ) -> bool:
        """
        Send document (PDF/EPUB) to Kindle via email.

        Args:
            kindle_email: Kindle email address.
            file_path: Path to the file.
            subject: Email subject.

        Returns:
            True if sent successfully.
        """
        doc_file = Path(file_path)
        if not doc_file.exists():
            logger.error(f"File not found: {file_path}")
            return False

        if not subject:
            subject = doc_file.stem
        
        # Use "Convert" subject line for EPUBs
        if doc_file.suffix.lower() == ".epub":
            subject = "Convert"
            
        subject = subject.replace("\n", " ").replace("\r", " ")

        try:
            masked = kindle_email[:3] + "***" + kindle_email[kindle_email.index("@"):] if "@" in kindle_email else "***"
            logger.info(f"Sending {doc_file.suffix[1:].upper()} to Kindle: {masked}")

            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = kindle_email
            msg["Subject"] = subject

            msg.attach(MIMEText("Morning Briefing", "plain"))

            with open(doc_file, "rb") as f:
                ext = doc_file.suffix.lower()
                subtype = "pdf" if ext == ".pdf" else "epub+zip" if ext == ".epub" else "octet-stream"
                attachment = MIMEApplication(f.read(), _subtype=subtype)
                attachment.add_header(
                    "Content-Disposition", "attachment", filename=doc_file.name
                )
                msg.attach(attachment)

            with self._connect_smtp() as server:
                server.send_message(msg)

            logger.info(f"{doc_file.suffix[1:].upper()} sent to Kindle: {masked}")
            return True

        except Exception as e:
            logger.error(f"Kindle send failed: {e}")
            return False

    def send_html_email(
        self,
        recipients: List[str],
        markdown_content: str,
        subject: Optional[str] = None,
        attachment_path: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send rich HTML briefing to a list of email addresses.

        Args:
            recipients: List of email addresses.
            markdown_content: Markdown content.
            subject: Email subject.
            attachment_path: Optional file to attach.

        Returns:
            Dictionary mapping email -> success boolean.
        """
        if not recipients:
            return {}

        if not subject:
            subject = "Atlas Morning Briefing"
        subject = subject.replace("\n", " ").replace("\r", " ")

        html_content = self._markdown_to_html(markdown_content)
        results = {}

        try:
            with self._connect_smtp() as server:
                for recipient in recipients:
                    try:
                        msg = MIMEMultipart("alternative")
                        msg["From"] = self.sender_email
                        msg["To"] = recipient
                        msg["Subject"] = subject

                        # Plain text fallback
                        plain_text = re.sub(r"[#*\[\]()]", "", markdown_content)
                        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
                        msg.attach(MIMEText(html_content, "html", "utf-8"))

                        # Optional attachment
                        if attachment_path:
                            doc_file = Path(attachment_path)
                            if doc_file.exists():
                                with open(doc_file, "rb") as f:
                                    ext = doc_file.suffix.lower()
                                    subtype = "pdf" if ext == ".pdf" else "epub+zip" if ext == ".epub" else "octet-stream"
                                    attachment = MIMEApplication(f.read(), _subtype=subtype)
                                    attachment.add_header(
                                        "Content-Disposition", "attachment", filename=doc_file.name
                                    )
                                    msg_mixed = MIMEMultipart("mixed")
                                    msg_mixed["From"] = msg["From"]; msg_mixed["To"] = msg["To"]; msg_mixed["Subject"] = msg["Subject"]
                                    msg_mixed.attach(msg); msg_mixed.attach(attachment)
                                    msg = msg_mixed

                        server.send_message(msg)
                        results[recipient] = True
                    except Exception as e:
                        logger.error(f"Failed to send to {recipient}: {e}")
                        results[recipient] = False
        except Exception as e:
            logger.error(f"SMTP connection failed: {e}")
            for r in recipients:
                if r not in results: results[r] = False

        return results

    def distribute(
        self,
        config: Dict,
        markdown_content: str,
        pdf_path: Optional[str] = None,
        epub_path: Optional[str] = None,
        subject: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, bool]:
        """Distribute briefing to all configured channels."""
        results = {}
        if dry_run: return results

        # Kindle (Prefer EPUB)
        kindle_email = config.get("kindle_email")
        if kindle_email:
            path = epub_path or pdf_path
            if path:
                results[f"kindle:{kindle_email}"] = self.send_kindle(kindle_email, path, subject)

        # Email list
        email_recipients = config.get("email_recipients", [])
        if email_recipients:
            res = self.send_html_email(email_recipients, markdown_content, subject, pdf_path)
            results.update(res)

        return results
