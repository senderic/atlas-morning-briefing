"""Shared pytest setup.

Some scanner modules (blog_scanner) import `feedparser`, which on certain
sandboxed CI environments fails to install cleanly because of its
sgmllib3k transitive dep. The worker code under test never actually
calls feedparser at runtime when scanners are mocked, so install a stub
module BEFORE pytest collects worker tests so import succeeds.

This is a no-op when feedparser is properly installed.
"""

import sys
import types


def _ensure_feedparser_importable():
    try:
        import feedparser  # noqa: F401
        return
    except ImportError:
        pass

    stub = types.ModuleType("feedparser")

    def _parse(*_args, **_kwargs):  # pragma: no cover - tests should mock
        raise RuntimeError(
            "feedparser stub: tests must mock BlogScanner; do not call parse()"
        )

    stub.parse = _parse
    sys.modules["feedparser"] = stub


_ensure_feedparser_importable()
