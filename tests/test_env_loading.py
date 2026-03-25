#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for environment variable expansion in config."""

import os
from unittest.mock import patch

import pytest
import yaml
from scripts.briefing_runner import load_config


class TestConfigEnvExpansion:
    def test_basic_expansion(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("api_key: ${TEST_API_KEY}")
        
        with patch.dict(os.environ, {"TEST_API_KEY": "secret123"}):
            config = load_config(str(config_file))
            assert config["api_key"] == "secret123"

    def test_expansion_with_default(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("email: ${TEST_EMAIL:-default@example.com}")
        
        # When env var is missing
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(str(config_file))
            assert config["email"] == "default@example.com"
            
        # When env var is present
        with patch.dict(os.environ, {"TEST_EMAIL": "user@domain.com"}):
            config = load_config(str(config_file))
            assert config["email"] == "user@domain.com"

    def test_expansion_in_list(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
recipients:
  - ${EMAIL_1}
  - ${EMAIL_2:-backup@mail.com}
""")
        
        env = {
            "EMAIL_1": "primary@mail.com"
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config(str(config_file))
            assert config["recipients"][0] == "primary@mail.com"
            assert config["recipients"][1] == "backup@mail.com"

    def test_complex_expansion(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
nested:
  key: "prefix-${ENV_PART}-suffix"
""")
        
        with patch.dict(os.environ, {"ENV_PART": "middle"}):
            config = load_config(str(config_file))
            assert config["nested"]["key"] == "prefix-middle-suffix"

    def test_comma_separated_emails(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
email_recipients:
  - ${RECIPIENT_EMAIL:-your-email@example.com}
""")
        
        with patch.dict(os.environ, {"RECIPIENT_EMAIL": "a@b.com,c@d.com"}):
            config = load_config(str(config_file))
            # It should still be a single string list item at this point
            assert config["email_recipients"] == ["a@b.com,c@d.com"]
