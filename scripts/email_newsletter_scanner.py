#!/usr/bin/env python3
"""Read configured newsletters from an IMAP mailbox without changing them."""

import base64
import email
import html
import imaplib
import logging
import re
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, parseaddr
from html.parser import HTMLParser
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from scripts.url_utils import normalize_url


logger = logging.getLogger(__name__)


class _NewsletterHTMLParser(HTMLParser):
    """Collect readable text and labeled links from newsletter HTML."""

    _BLOCK_TAGS = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "li", "p", "section", "table", "td", "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self.headings: List[str] = []
        self._heading: Optional[List[str]] = None
        self._skip_depth = 0
        self._link: Optional[List[Any]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4"}:
            self._heading = []
        if tag == "a":
            self._link = [dict(attrs).get("href") or "", []]

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._link is not None:
            label = " ".join("".join(self._link[1]).split())
            self.links.append((label, html.unescape(self._link[0])))
            self._link = None
        if tag in {"h1", "h2", "h3", "h4"} and self._heading is not None:
            self.headings.append(" ".join("".join(self._heading).split()))
            self._heading = None
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._heading is not None:
            self._heading.append(data)
        if self._link is not None:
            self._link[1].append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


class EmailNewsletterScanner:
    """Fetch recent messages from explicitly configured newsletter senders."""

    def __init__(
        self,
        config: Dict[str, Any],
        username: str,
        password: str,
        imap_factory: Callable[..., Any] = imaplib.IMAP4_SSL,
    ) -> None:
        self.config = config
        self.username = username
        self.password = password
        self.imap_factory = imap_factory

    @staticmethod
    def _decode_header(value: Optional[str]) -> str:
        return str(make_header(decode_header(value or "")))

    @staticmethod
    def _clean_summary(text: str, source_config: Dict[str, Any]) -> str:
        """Remove email-client padding, private addresses, and configured footers."""
        text = re.sub(
            "[\u00ad\u034f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]",
            "",
            text,
        )
        lowered = text.lower()
        footer_indexes = [
            lowered.find(str(marker).lower())
            for marker in source_config.get("footer_markers", [])
            if str(marker).strip() and lowered.find(str(marker).lower()) >= 0
        ]
        if footer_indexes:
            text = text[:min(footer_indexes)]
        text = re.sub(
            r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])",
            "[redacted-email]",
            text,
        )
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _decode_tracking_url(url: str) -> str:
        """Decode known newsletter redirects without making a tracking request."""
        def display_url(value: str) -> str:
            canonical = normalize_url(value)
            if canonical.startswith("//"):
                return f"{urlparse(value).scheme}:{canonical}"
            return canonical

        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        base64_click_hosts = {
            "link.axios.com",
            "link.newsletters.sandiegouniontribune.com",
            "linkst.newsletters.sandiegouniontribune.com",
        }
        if (
            parsed.netloc in base64_click_hosts
            and len(parts) >= 3
            and parts[0] == "click"
        ):
            encoded = parts[2]
            try:
                padding = "=" * (-len(encoded) % 4)
                destination = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
                if destination.startswith(("https://", "http://")):
                    return display_url(destination)
            except (ValueError, UnicodeDecodeError):
                pass
        if (
            parsed.netloc == "tracking.tldrnewsletter.com"
            and len(parts) >= 2
            and parts[0] == "CL0"
        ):
            destination = unquote(parts[1])
            if destination.startswith(("https://", "http://")):
                return display_url(destination)
        return display_url(url)

    @classmethod
    def _preferred_link(
        cls,
        links: List[Tuple[str, str]],
        source_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        source_config = source_config or {}
        for label, url in links:
            if label.strip().lower() in {"view in browser", "view online"}:
                decoded = cls._decode_tracking_url(url)
                if not cls._is_opaque_tracking_link(decoded):
                    return decoded
        preferred_domains = {
            str(domain).lower().removeprefix("www.")
            for domain in source_config.get("preferred_link_domains", [])
        }
        if preferred_domains:
            for _, url in links:
                decoded = cls._decode_tracking_url(url)
                if cls._is_opaque_tracking_link(decoded):
                    continue
                hostname = (urlparse(decoded).hostname or "").lower()
                if any(
                    hostname == domain or hostname.endswith(f".{domain}")
                    for domain in preferred_domains
                ):
                    return decoded
        fallback = str(source_config.get("fallback_link") or "").strip()
        return cls._decode_tracking_url(fallback) if fallback else ""

    @staticmethod
    def _message_content(message: Any) -> Tuple[str, List[Tuple[str, str]]]:
        html_body = ""
        plain_body = ""

        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            content_type = part.get_content_type()
            if content_type not in {"text/html", "text/plain"}:
                continue
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
            if content_type == "text/html" and not html_body:
                html_body = decoded
            elif content_type == "text/plain" and not plain_body:
                plain_body = decoded

        links: List[Tuple[str, str]] = []
        if html_body:
            parser = _NewsletterHTMLParser()
            parser.feed(html_body)
            summary = parser.text()
            links = parser.links
        else:
            summary = "\n".join(
                " ".join(line.split()) for line in plain_body.splitlines() if line.strip()
            )

        return summary, links

    def parse_message(
        self,
        raw_message: bytes,
        source_name: str,
        source_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        message = email.message_from_bytes(raw_message)
        summary, links = self._message_content(message)
        source_config = source_config or {}
        summary = self._clean_summary(summary, source_config)

        max_chars = int(source_config.get("max_chars", self.config.get("max_chars", 12_000)))
        summary = summary[:max_chars].strip()
        date_value = ""
        try:
            parsed_date = parsedate_to_datetime(message.get("Date", ""))
            if parsed_date:
                date_value = parsed_date.isoformat()
        except (TypeError, ValueError):
            pass

        from_name, _ = parseaddr(self._decode_header(message.get("From")))
        return {
            "source": source_name,
            "title": self._decode_header(message.get("Subject")).strip(),
            "link": self._preferred_link(links, source_config),
            "summary": summary,
            "published": date_value,
            "author": from_name,
            "message_id": (message.get("Message-ID") or "").strip(),
            "input_type": "email_newsletter",
        }

    @staticmethod
    def _is_opaque_tracking_link(url: str) -> bool:
        hostname = (urlparse(url).hostname or "").lower()
        return hostname in {
            "c.gle",
            "go.sparkpostmail.com",
            "link.mail.beehiiv.com",
            "links.tldrnewsletter.com",
            "tracking.tldrnewsletter.com",
            "link.newsletters.sandiegouniontribune.com",
            "linkst.newsletters.sandiegouniontribune.com",
        } or hostname.endswith(".prospect2.com")

    @staticmethod
    def _domain_allowed(url: str, allowed_domains: set) -> bool:
        if not allowed_domains:
            return True
        hostname = (urlparse(url).hostname or "").lower()
        return any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in allowed_domains
        )

    def parse_articles(
        self, raw_message: bytes, source_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return one newsletter item or opt-in story-level items."""
        source_name = str(source_config.get("name") or "Email newsletter").strip()
        base_article = self.parse_message(
            raw_message, source_name, source_config=source_config
        )
        if source_config.get("mode") == "sections":
            message = email.message_from_bytes(raw_message)
            parser = _NewsletterHTMLParser()
            for part in message.walk():
                if part.get_content_type() == "text/html" and part.get_content_disposition() != "attachment":
                    parser.feed((part.get_payload(decode=True) or b"").decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    ))
                    break
            return self.split_sections(base_article, source_config, parser.headings)
        if source_config.get("mode") != "stories":
            return [base_article]

        message = email.message_from_bytes(raw_message)
        full_summary, links = self._message_content(message)
        full_summary = self._clean_summary(full_summary, source_config)
        lines = [line.strip() for line in full_summary.splitlines() if line.strip()]
        title_pattern = str(source_config.get("story_title_pattern") or "").strip()
        excluded_terms = {
            "advertise", "manage preferences", "privacy", "read more",
            "share this story", "sign up", "sponsor", "subscribe",
            "unsubscribe", "view in browser", "view online",
        }
        excluded_terms.update(
            str(term).strip().lower()
            for term in source_config.get("exclude_title_terms", [])
            if str(term).strip()
        )
        allowed_domains = {
            str(domain).lower().removeprefix("www.")
            for domain in source_config.get("story_link_domains", [])
        }
        max_stories = max(1, int(source_config.get("max_stories", 5)))

        candidates = []
        seen_links = set()
        for original_label, raw_url in links:
            label = " ".join(original_label.split()).strip()
            lowered = label.lower()
            if not label or any(term in lowered for term in excluded_terms):
                continue
            if title_pattern and not re.search(title_pattern, label, flags=re.IGNORECASE):
                continue
            if not title_pattern and len(label) < 20:
                continue
            decoded_url = self._decode_tracking_url(raw_url)
            if (
                not decoded_url
                or urlparse(decoded_url).scheme not in {"http", "https"}
                or self._is_opaque_tracking_link(decoded_url)
            ):
                continue
            if not self._domain_allowed(decoded_url, allowed_domains):
                continue
            canonical = normalize_url(decoded_url)
            if not canonical or canonical in seen_links:
                continue
            seen_links.add(canonical)
            title = (
                re.sub(title_pattern, "", label, flags=re.IGNORECASE).strip()
                if title_pattern
                else label
            )
            title = title.strip(" -|–—")
            if title:
                candidates.append((label, title, decoded_url))
            if len(candidates) >= max_stories:
                break

        if not candidates:
            return [base_article]

        boundary_labels = {
            " ".join(label.split()).strip()
            for label, _ in links
            if len(" ".join(label.split()).strip()) >= 20
        }
        boundary_indexes = [
            index for index, line in enumerate(lines) if line in boundary_labels
        ]
        summary_chars = max(100, int(source_config.get("story_summary_chars", 1_200)))
        articles = []
        cursor = 0
        for index, (original_label, title, link) in enumerate(candidates):
            try:
                start = lines.index(original_label, cursor)
            except ValueError:
                start = -1
            if start >= 0:
                end = next(
                    (boundary for boundary in boundary_indexes if boundary > start),
                    len(lines),
                )
                story_summary = "\n".join(lines[start + 1:end]).strip()
                cursor = start + 1
            else:
                story_summary = ""
            if not story_summary:
                story_summary = base_article["summary"]
            article = base_article.copy()
            article.update(
                {
                    "title": title,
                    "link": link,
                    "summary": story_summary[:summary_chars].strip(),
                    "message_id": f"{base_article.get('message_id', '')}#story-{index + 1}",
                    "newsletter_story": True,
                }
            )
            articles.append(article)
        return articles

    @classmethod
    def split_sections(
        cls, article: Dict[str, Any], source_config: Dict[str, Any],
        headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Split HTML headings or numbered digest sections, including saved input."""
        lines = str(article.get("summary", "")).splitlines()
        pattern = source_config.get("section_title_pattern")
        labels = set(headings or [])
        # Saved snapshots contain text, not HTML. Repeated table-of-contents
        # labels reveal the headings without a dependency on one issue's topics.
        if not labels and not pattern:
            labels = {line for line in lines[:20] if 20 <= len(line) <= 200 and lines.count(line) > 1}
        excluded = [str(t).lower() for t in source_config.get("exclude_title_terms", [])]
        starts = []
        for index, line in enumerate(lines):
            if (pattern and re.search(pattern, line)) or (not pattern and line in labels):
                if not pattern and line in lines[index + 1:]:
                    continue  # table of contents, rather than the actual story
                starts.append((index, line))
        result = []
        for position, (start, heading) in enumerate(starts):
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            if any(term in heading.lower() for term in excluded):
                continue
            title = re.sub(pattern, "", heading).strip() if pattern else heading
            body = "\n".join(lines[start + 1:end]).strip()
            section_config = {"footer_markers": source_config.get("section_footer_markers", [])}
            body = cls._clean_summary(body, section_config)
            if not body:
                continue
            result.append({
                **article,
                "title": title,
                "summary": body[:int(source_config.get("story_summary_chars", 2400))],
                "message_id": f"{article.get('message_id', '')}#section-{position + 1}",
                "newsletter_section": True,
                "original_email_title": article.get("title", ""),
            })
            if len(result) >= int(source_config.get("max_stories", 6)):
                break
        return result or [article]

    def scan(self) -> List[Dict[str, Any]]:
        sources = self.config.get("sources") or []
        if not sources or not self.username or not self.password:
            return []

        host = self.config.get("host", "imap.gmail.com")
        port = int(self.config.get("port", 993))
        mailbox = self.config.get("mailbox", "INBOX")
        lookback_days = max(1, int(self.config.get("lookback_days", 2)))
        max_messages = max(1, int(self.config.get("max_messages", 3)))
        max_articles = max(1, int(self.config.get("max_articles", 1_000)))
        articles: List[Dict[str, Any]] = []
        source_batches: List[List[Dict[str, Any]]] = []
        seen_ids = set()
        seen_links = set()

        try:
            with self.imap_factory(host, port) as connection:
                connection.login(self.username, self.password)
                status, _ = connection.select(mailbox, readonly=True)
                if status != "OK":
                    raise RuntimeError(f"could not open mailbox {mailbox!r} read-only")

                for source in sources:
                    source_articles: List[Dict[str, Any]] = []
                    source_batches.append(source_articles)
                    name = str(source.get("name") or "Email newsletter").strip()
                    sender = str(source.get("from") or "").strip()
                    if not sender:
                        continue
                    query = f'"newer_than:{lookback_days}d from:{sender}"'
                    status, data = connection.uid("SEARCH", None, "X-GM-RAW", query)
                    if status != "OK" or not data:
                        logger.warning("Newsletter search failed for %s", name)
                        continue

                    source_max_messages = max(1, int(source.get("max_messages", max_messages)))
                    for uid in reversed(data[0].split()[-source_max_messages:]):
                        status, parts = connection.uid("FETCH", uid, "(BODY.PEEK[])")
                        if status != "OK":
                            continue
                        raw = next(
                            (part[1] for part in parts if isinstance(part, tuple)), b""
                        )
                        if not raw:
                            continue
                        exclude_subject = source.get("exclude_subject_pattern")
                        subject = self._decode_header(email.message_from_bytes(raw).get("Subject"))
                        if exclude_subject and re.search(exclude_subject, subject, re.IGNORECASE):
                            continue
                        for article in self.parse_articles(raw, source):
                            message_id = article.get("message_id")
                            canonical_link = normalize_url(article.get("link", ""))
                            fallback_link = normalize_url(str(source.get("fallback_link") or ""))
                            if canonical_link and (article.get("newsletter_section") or canonical_link == fallback_link):
                                canonical_link += "#" + article["title"].lower()
                            if message_id and message_id in seen_ids:
                                continue
                            if canonical_link and canonical_link in seen_links:
                                continue
                            if message_id:
                                seen_ids.add(message_id)
                            if canonical_link:
                                seen_links.add(canonical_link)
                            if article.get("title") and article.get("summary"):
                                source_articles.append(article)
        except Exception as exc:
            logger.error("Email newsletter scan failed: %s", exc)
            return []

        # Round robin across senders before applying the global budget: a
        # daily digest must not consume every slot before civic mail is read.
        for row in zip_longest(*source_batches):
            for article in row:
                if article is not None and len(articles) < max_articles:
                    articles.append(article)
        logger.info("Found %d email newsletter stories", len(articles))
        return articles
