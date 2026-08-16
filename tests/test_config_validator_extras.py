# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Extra tests for config_validator to cover branches missed by the base file."""

import pytest

from scripts.config_validator import validate_config, check_environment


class TestValidateExtras:
    def test_int_field_out_of_range_warns_only(self):
        cfg = {"arxiv_topics": ["X"], "arxiv_days_back": 999}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is True  # warning, not error
        assert any("recommended range" in m for m in msgs)

    def test_blog_feeds_not_list(self):
        cfg = {"arxiv_topics": ["X"], "blog_feeds": "string"}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any("blog_feeds" in m for m in msgs)

    def test_stocks_not_list(self):
        cfg = {"arxiv_topics": ["X"], "stocks": "AAPL"}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False

    def test_news_queries_not_list(self):
        cfg = {"arxiv_topics": ["X"], "news_queries": "AI"}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False

    def test_interest_graph_valid(self):
        cfg = {
            "arxiv_topics": ["X"],
            "interest_graph": {
                "max_dynamic_queries": 2,
                "roots": [
                    {"id": "r1", "query": "defense", "children": [{"id": "l1", "query": "palantir"}]}
                ],
            },
        }
        is_valid, msgs = validate_config(cfg)
        assert is_valid is True
        assert msgs == []

    def test_interest_graph_empty_roots(self):
        cfg = {"arxiv_topics": ["X"], "interest_graph": {"roots": []}}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any("interest_graph.roots" in m for m in msgs)

    def test_interest_graph_node_missing_query(self):
        cfg = {
            "arxiv_topics": ["X"],
            "interest_graph": {"roots": [{"id": "r1"}]},
        }
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any(".query" in m for m in msgs)

    def test_interest_graph_duplicate_ids(self):
        cfg = {
            "arxiv_topics": ["X"],
            "interest_graph": {
                "roots": [
                    {"id": "r1", "query": "a"},
                    {"id": "r1", "query": "b"},
                ]
            },
        }
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any("duplicate" in m for m in msgs)

    def test_interest_graph_bad_max_dynamic(self):
        cfg = {
            "arxiv_topics": ["X"],
            "interest_graph": {"max_dynamic_queries": "many", "roots": [{"id": "r", "query": "x"}]},
        }
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any("max_dynamic_queries" in m for m in msgs)

    def test_kindle_email_without_kindle_warns(self):
        cfg = {"arxiv_topics": ["X"], "kindle_email": "test@example.com"}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is True  # warning only
        assert any("kindle" in m for m in msgs)

    def test_kindle_placeholder_no_warn(self):
        cfg = {"arxiv_topics": ["X"], "kindle_email": "YOUR_NAME@kindle.com"}
        is_valid, _ = validate_config(cfg)
        assert is_valid is True

    def test_pdf_font_size_not_number(self):
        cfg = {"arxiv_topics": ["X"], "pdf": {"font_size": "ten"}}
        is_valid, msgs = validate_config(cfg)
        assert is_valid is False
        assert any("font_size" in m for m in msgs)

    def test_pdf_line_spacing_not_number(self):
        cfg = {"arxiv_topics": ["X"], "pdf": {"line_spacing": "wide"}}
        is_valid, _ = validate_config(cfg)
        assert is_valid is False

    def test_bedrock_models_not_dict(self):
        cfg = {"arxiv_topics": ["X"], "bedrock": {"models": "not a dict"}}
        is_valid, _ = validate_config(cfg)
        assert is_valid is False

    def test_unknown_tier_warns(self):
        cfg = {
            "arxiv_topics": ["X"],
            "bedrock": {"models": {"ultra": "some-model"}},
        }
        is_valid, msgs = validate_config(cfg)
        assert is_valid is True  # warning only
        assert any("recognized tier" in m for m in msgs)


class TestCheckEnvExtras:
    def test_gmail_present_no_warning(self, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "u@x.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
        cfg = {}
        warnings = check_environment(cfg, dry_run=False)
        assert not any("GMAIL" in w for w in warnings)
