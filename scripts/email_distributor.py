#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Email distribution module.

Sends briefings via multiple channels:
- Kindle: PDF attachment (Kindle only supports PDF/MOBI)
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

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
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
        # (LLM outputs, news titles, blog titles may contain malicious markup)
        if HAS_NH3:
            html_body = nh3.clean(html_body)
        else:
            logger.warning("nh3 not installed; HTML email output is not sanitized")

        # Handle [RIGHT] markers for alignment — applied after sanitization so
        # the injected style attribute is not stripped by nh3.
        html_body = html_body.replace("[RIGHT]", '<span style="float: right;">').replace("[/RIGHT]", "</span>")

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
  h4 {{
    color: #57606a;
    font-size: 12px;
    font-weight: 600;
    margin-top: 16px;
    margin-bottom: 2px;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }}
  h4 + p {{
    font-size: 13px;
    color: #57606a;
    margin-top: 0;
    line-height: 1.4;
  }}
  p {{
    margin: 8px 0;
    font-size: 14px;
  }}
  em {{
    color: #57606a;
  }}
  strong {{
    color: #0d1117;
  }}
  a {{
    color: #1f6feb;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  ul, ol {{
    padding-left: 24px;
    font-size: 14px;
  }}
  li {{
    margin: 4px 0;
  }}
  code {{
    background-color: #f0f3f6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'SFMono-Regular', Consolas, monospace;
  }}
  pre {{
    background-color: #0d1117;
    color: #e6edf3;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: inherit;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 13px;
  }}
  th, td {{
    border: 1px solid #d0d7de;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background-color: #f0f3f6;
    font-weight: 600;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e1e4e8;
    margin: 20px 0;
  }}
  .footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e1e4e8;
    font-size: 12px;
    color: #8b949e;
    text-align: center;
  }}
  /* Stock colors */
  .stock-up {{ color: #1a7f37; font-weight: 600; }}
  .stock-down {{ color: #cf222e; font-weight: 600; }}
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
        Send document to Kindle via email.

        Kindle supports PDF, EPUB, and MOBI.

        Args:
            kindle_email: Kindle email address.
            file_path: Path to the document file (PDF or EPUB).
            subject: Email subject (defaults to filename).

        Returns:
            True if sent successfully.
        """
        doc_file = Path(file_path)
        if not doc_file.exists():
            logger.error(f"Document file not found: {file_path}")
            return False

        if not subject:
            subject = doc_file.stem
        
        # Use "Convert" subject line for EPUBs sent to Kindle.
        # This triggers Amazon's server-side conversion path which is more 
        # robust against minor EPUB structural issues (E999 errors).
        if doc_file.suffix.lower() == ".epub":
            logger.info("Using 'Convert' subject line for EPUB Kindle delivery")
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
                if ext == ".pdf":
                    subtype = "pdf"
                    maintype = "application"
                elif ext == ".epub":
                    subtype = "epub+zip"
                    maintype = "application"
                else:
                    subtype = "octet-stream"
                    maintype = "application"

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
            markdown_content: Markdown briefing content.
            subject: Email subject.
            attachment_path: Optional PDF or EPUB to attach alongside HTML.

        Returns:
            Dictionary mapping email -> success boolean.
        """
        if not recipients:
            logger.warning("No email recipients configured")
            return {}

        if not subject:
            subject = "Atlas Morning Briefing"
        subject = subject.replace("\n", " ").replace("\r", " ")

        html_content = self._markdown_to_html(markdown_content)
        results = {}

        try:
            with self._connect_smtp() as server:
                msg = MIMEMultipart("alternative")
                msg["From"] = self.sender_email
                msg["To"] = ", ".join(recipients)
                msg["Subject"] = subject

                # Plain text fallback (stripped markdown)
                plain_text = re.sub(r"[#*\[\]()]", "", markdown_content)
                msg.attach(MIMEText(plain_text, "plain", "utf-8"))

                # Rich HTML version
                msg.attach(MIMEText(html_content, "html", "utf-8"))

                # Optional attachment
                if attachment_path:
                    doc_file = Path(attachment_path)
                    if doc_file.exists():
                        with open(doc_file, "rb") as f:
                            ext = doc_file.suffix.lower()
                            subtype = "pdf" if ext == ".pdf" else "epub+zip" if ext == ".epub" else "octet-stream"
                            
                            attachment = MIMEApplication(
                                f.read(), _subtype=subtype
                            )
                            attachment.add_header(
                                "Content-Disposition",
                                "attachment",
                                filename=doc_file.name,
                            )
                            # Switch to mixed for attachment support
                            msg_with_attach = MIMEMultipart("mixed")
                            msg_with_attach["From"] = msg["From"]
                            msg_with_attach["To"] = msg["To"]
                            msg_with_attach["Subject"] = msg["Subject"]
                            msg_with_attach.attach(msg)
                            msg_with_attach.attach(attachment)
                            msg = msg_with_attach

                server.send_message(msg)
                
                # Update results for all recipients
                for recipient in recipients:
                    masked_r = recipient[:3] + "***" + recipient[recipient.index("@"):] if "@" in recipient else "***"
                    logger.info(f"HTML briefing sent to: {masked_r}")
                    results[recipient] = True

        except Exception as e:
            logger.error(f"Failed to send HTML email: {e}")
            for recipient in recipients:
                results[recipient] = False

        sent = sum(1 for v in results.values() if v)
        logger.info(f"Email distribution: {sent}/{len(recipients)} sent successfully")
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
        """
        Distribute briefing to all configured channels.

        Args:
            config: Distribution config with 'kindle_email' and 'email_recipients'.
            markdown_content: Markdown briefing content.
            pdf_path: Path to generated PDF.
            epub_path: Path to generated EPUB.
            subject: Email subject.
            dry_run: If True, skip actual sending.

        Returns:
            Dictionary mapping channel/email -> success boolean.
        """
        results = {}

        if dry_run:
            logger.info("Dry run: skipping all email distribution")
            return results

        # Kindle (Prefer EPUB for reflowable text if available, else PDF)
        kindle_email = config.get("kindle_email")
        if kindle_email:
            if epub_path:
                results[f"kindle:{kindle_email}"] = self.send_kindle(
                    kindle_email, epub_path, subject
                )
            elif pdf_path:
                results[f"kindle:{kindle_email}"] = self.send_kindle(
                    kindle_email, pdf_path, subject
                )

        # Email list (HTML)
        email_recipients_raw = config.get("email_recipients", [])
        email_recipients_set = set()
        for r in email_recipients_raw:
            if isinstance(r, str) and "," in r:
                for email in r.split(","):
                    if email.strip():
                        email_recipients_set.add(email.strip())
            elif r:
                email_recipients_set.add(r)
        
        email_recipients = list(email_recipients_set)

        if email_recipients:
            # For regular email, PDF is preferred. Don't send EPUB to non-Kindle.
            attachment = pdf_path
            html_results = self.send_html_email(
                recipients=email_recipients,
                markdown_content=markdown_content,
                subject=subject,
                attachment_path=attachment,
            )
            results.update(html_results)

        return results

