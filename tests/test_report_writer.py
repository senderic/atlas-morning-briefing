"""Behavior tests for the report-writing routing boundary."""

from scripts.report_writer import ReportWriter


class _Client:
    """Small in-memory client: only the external process is substituted."""

    def __init__(self, response=None, available=True, model="test-model"):
        self.response = response
        self.available = available
        self.model = model
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response

    def get_usage_summary(self, *args, **kwargs):
        return "UNDERLYING CLIENT USAGE"


class TestReportWriter:
    def test_codex_prose_is_served_without_fallback(self):
        """Catches a regression where a good Codex result still bills the chain."""
        codex = _Client("Reader-ready prose.", model="codex-model")
        fallback = _Client("Fallback prose.")

        writer = ReportWriter(codex, fallback)

        assert writer.invoke("write this", tier="medium") == "Reader-ready prose."
        assert fallback.calls == []
        assert writer.last_backend == "codex"
        assert writer.fallback_count == 0

    def test_leaked_codex_prose_uses_heavy_fallback(self):
        """Catches a regression where rationale reaches readers or the wrong tier runs."""
        codex = _Client("Let me think through what the user wants here.")
        fallback = _Client("Clean report prose.")

        writer = ReportWriter(codex, fallback)

        assert writer.invoke("write this", tier="light") == "Clean report prose."
        assert len(fallback.calls) == 1
        assert fallback.calls[0][0] == "write this"
        assert fallback.calls[0][1]["tier"] == "heavy"
        assert writer.last_backend == "fallback"
        assert writer.fallback_count == 1

    def test_available_fallback_writes_when_codex_is_unavailable(self):
        """Catches tying writer availability to the optional Codex executable."""
        codex = _Client(available=False)
        fallback = _Client("Fallback-only report.")

        writer = ReportWriter(codex, fallback)

        assert writer.available is True
        assert writer.invoke("write this") == "Fallback-only report."
        assert writer.last_backend == "fallback"
        assert writer.fallback_count == 1

    def test_each_call_retries_codex_after_an_earlier_fallback(self):
        """Catches a failed first call permanently pinning later prose to fallback."""
        codex = _Client(model="codex-model")
        codex.response = iter([None, "Second call from Codex."])
        codex.invoke = lambda prompt, **kwargs: (
            codex.calls.append((prompt, kwargs)) or next(codex.response)
        )
        fallback = _Client("First call from fallback.")
        writer = ReportWriter(codex, fallback)

        assert writer.invoke("first report") == "First call from fallback."
        assert writer.invoke("second report") == "Second call from Codex."
        assert len(codex.calls) == 2
        assert len(fallback.calls) == 1
        assert writer.backends == ["fallback", "codex"]
        assert writer.fallback_count == 1

    def test_usage_summary_describes_routing_without_underlying_usage(self):
        """Catches footer duplication of the CompositeClient's own usage lines."""
        codex = _Client(None, available=False, model="codex-model")
        fallback = _Client("Fallback-only report.")
        writer = ReportWriter(codex, fallback)
        writer.invoke("write this")

        summary = writer.get_usage_summary()

        assert "Report writer" in summary
        assert "1 fallback" in summary
        assert "UNDERLYING CLIENT USAGE" not in summary
