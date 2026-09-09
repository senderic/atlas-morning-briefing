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

try:
    from premailer import transform
    HAS_PREMAILER = True
except ImportError:
    HAS_PREMAILER = False
    def transform(html: str, **kwargs) -> str:
        return html

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Gmail clips messages larger than 102KB
GMAIL_CLIP_LIMIT_KB = 102
GMAIL_WARN_LIMIT_KB = 90


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
        Convert markdown briefing to responsive HTML email.

        Optimized for the Gmail mobile app, which shrinks the whole message to
        fit the screen when any element is wider than the viewport — the reason
        a 14px font can arrive looking like 8px. The template therefore has no
        fixed pixel widths and forces long tokens (model IDs, URLs) to wrap.

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
        # the injected style attribute is not stripped by nh3. Floats are
        # unreliable in email clients, so this only nudges the byline right.
        html_body = html_body.replace("[RIGHT]", '<span style="float: right;">').replace("[/RIGHT]", "</span>")

        html = self._wrap_in_template(html_body)

        # Inline the CSS. Gmail with a Google account honours the <style> block,
        # but Gmail for non-Google accounts, Outlook and Yahoo strip it, so both
        # forms ship. keep_style_tags keeps the media queries alive; classes are
        # kept for the same reason (remove_classes would break them).
        if HAS_PREMAILER:
            html = transform(
                html,
                base_url=None,
                keep_style_tags=True,
                strip_important=False,
                remove_classes=False,
                exclude_pseudoclasses=":hover, :active, :focus, :visited",
            )
        else:
            logger.warning("premailer not installed; CSS not inlined for email clients")

        # Warn if HTML size approaches Gmail's clip limit
        size_kb = len(html.encode("utf-8")) / 1024
        if size_kb > GMAIL_WARN_LIMIT_KB:
            logger.warning(
                f"HTML email size: {size_kb:.1f}KB (Gmail clips at {GMAIL_CLIP_LIMIT_KB}KB)"
            )

        return html

    def _wrap_in_template(self, html_body: str) -> str:
        """
        Wrap the converted markdown in a fluid, table-based email shell.

        Rules this template follows, in priority order:

        1. No element may be wider than the screen. The Gmail app scales the
           entire message down to fit its widest element, so one 577px table
           shrinks every paragraph with it. Hence width:100%/max-width instead
           of a fixed 600px, table-layout:fixed, and word-break on code spans
           (the usage tables carry 48-character model IDs that cannot wrap).
        2. min-width:100% on the outer table, which is what stops the Gmail
           Android app "munging" the layout narrower still.
        3. 16px base text, 15px in tables — under ~14px Gmail may re-scale text
           on its own, and this is read on a phone at arm's length.
        4. Layout in tables, not divs; only properties in Gmail's supported CSS
           list; padding, never margin, for structural spacing.
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="format-detection" content="telephone=no">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<style>
  /* Inlined by premailer where possible; the block itself is kept so that
     the media query below still applies in clients that honour <style>. */
  body {{
    margin: 0;
    padding: 0;
    width: 100% !important;
    background-color: #f6f8fa;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #1a1a1a;
    -webkit-text-size-adjust: 100%;
    -ms-text-size-adjust: 100%;
  }}
  .wrapper {{
    width: 100%;
    min-width: 100%;
    background-color: #f6f8fa;
  }}
  .gutter {{
    padding: 16px 8px;
  }}
  .container {{
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
    background-color: #ffffff;
    border: 1px solid #d8dee4;
  }}
  .content {{
    padding: 20px;
  }}
  h1 {{
    color: #0d1117;
    font-size: 24px;
    font-weight: 700;
    line-height: 1.25;
    border-bottom: 2px solid #58a6ff;
    padding-bottom: 8px;
    margin: 0 0 12px 0;
  }}
  h2 {{
    color: #1f6feb;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.3;
    border-bottom: 1px solid #e1e4e8;
    padding-bottom: 6px;
    margin: 28px 0 12px 0;
  }}
  h3 {{
    color: #24292f;
    font-size: 17px;
    font-weight: 600;
    line-height: 1.35;
    margin: 22px 0 6px 0;
  }}
  h4 {{
    color: #57606a;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 16px 0 4px 0;
  }}
  p {{
    margin: 0 0 12px 0;
    font-size: 16px;
    line-height: 1.6;
  }}
  h4 + p {{
    font-size: 15px;
    color: #444d56;
    margin-top: 0;
  }}
  em {{
    color: #57606a;
  }}
  strong {{
    color: #0d1117;
    font-weight: 700;
  }}
  a {{
    color: #0969da;
    text-decoration: underline;
    word-break: break-word;
  }}
  ul, ol {{
    padding-left: 22px;
    margin: 0 0 12px 0;
  }}
  li {{
    font-size: 16px;
    line-height: 1.55;
    margin: 0 0 6px 0;
  }}
  /* Tables must never set the message width — see rule 1 above. */
  /* Auto layout, not table-layout:fixed: fixed gives every column an equal
     share, which is narrow enough that ordinary words ("Semiconductor") break
     mid-word. Auto plus the break rules below fits the width without that. */
  table {{
    width: 100%;
    max-width: 100%;
    border-collapse: collapse;
    margin: 0 0 14px 0;
  }}
  /* Only the properties that keep a table from setting the message width are
     inlined per cell; a briefing can carry 200+ cells and inlining the full
     rule on each one added ~30KB, pushing the message toward Gmail's 102KB
     clip limit. The decoration lives in the non-inlined block below. */
  th, td {{
    font-size: 15px;
    padding: 7px 8px;
    /* The dense diagnostic tables (6 columns of model IDs and token counts)
       have a min-content width of ~460px, so cells must be allowed to break
       inside a word — otherwise the table sets the message width and the Gmail
       app shrinks the entire briefing to fit it. Auto table layout keeps this
       from firing on the roomier tables. */
    word-break: break-word;
  }}
  code {{
    background-color: #f0f3f6;
    padding: 1px 4px;
    font-size: 14px;
    font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
    /* Model IDs are 45+ characters with no spaces; without this they set a
       floor on the message width and the whole email shrinks. */
    word-break: break-all;
    overflow-wrap: anywhere;
  }}
  pre {{
    background-color: #0d1117;
    color: #e6edf3;
    padding: 14px;
    font-size: 13px;
    font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
    white-space: pre-wrap;
    word-break: break-all;
    margin: 0 0 14px 0;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: inherit;
    font-size: 13px;
  }}
  blockquote {{
    border-left: 3px solid #d0d7de;
    padding: 2px 0 2px 12px;
    margin: 0 0 12px 0;
    color: #57606a;
  }}
  hr {{
    border: 0;
    border-top: 1px solid #e1e4e8;
    margin: 20px 0;
  }}
  img {{
    max-width: 100%;
    height: auto;
  }}
  .footer {{
    margin-top: 28px;
    padding-top: 12px;
    border-top: 1px solid #e1e4e8;
    font-size: 13px;
    color: #6e7781;
    text-align: center;
  }}
  .stock-up {{ color: #1a7f37; font-weight: 700; }}
  .stock-down {{ color: #cf222e; font-weight: 700; }}
</style>
<style data-premailer="ignore">
  /* Left in the <style> block on purpose — Gmail with a Google account honours
     it, and inlining any of this per cell is what blew up the message size. */
  th, td {{
    line-height: 1.4;
    text-align: left;
    vertical-align: top;
    border-bottom: 1px solid #e1e4e8;
  }}
  th {{
    background-color: #f6f8fa;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: #57606a;
    /* Headers are short; let them size the column rather than wrap mid-word. */
    overflow-wrap: normal;
  }}

  /* Phones: reclaim the horizontal padding and tighten the dense tables. */
  @media only screen and (max-width: 600px) {{
    .gutter {{ padding: 8px 4px !important; }}
    .content {{ padding: 16px 14px !important; }}
    h1 {{ font-size: 22px !important; }}
    h2 {{ font-size: 19px !important; }}
    th, td {{ font-size: 14px !important; padding: 6px 5px !important; }}
    th {{ font-size: 12px !important; }}
    code {{ font-size: 13px !important; }}
  }}
</style>
</head>
<body>
<table role="presentation" class="wrapper" width="100%" cellpadding="0" cellspacing="0" border="0">
  <tr>
    <td class="gutter" align="center">
      <table role="presentation" class="container" width="100%" cellpadding="0" cellspacing="0" border="0" align="center">
        <tr>
          <td class="content">
{html_body}
            <div class="footer">
              Atlas Morning Briefing<br>
              <a href="https://github.com/senderic/atlas-morning-briefing">GitHub</a>
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
</body>
</html>"""

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
