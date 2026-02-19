"""Tests for configuration loading and saving."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vocab_trainer.config import DEFAULTS, Settings, load_settings, save_settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.llm_provider == "ollama"
        assert s.session_size == 20
        assert s.tts_provider == "edge-tts"

    def test_to_dict(self):
        s = Settings()
        d = s.to_dict()
        assert d["llm_provider"] == "ollama"
        assert isinstance(d["vocab_files"], list)
        assert len(d) == 13  # all fields present

    def test_to_dict_roundtrip(self):
        s = Settings(llm_provider="anthropic", session_size=30)
        d = s.to_dict()
        s2 = Settings(**d)
        assert s2.llm_provider == "anthropic"
        assert s2.session_size == 30

    def test_custom_values(self):
        s = Settings(llm_provider="anthropic", llm_model="claude-3", session_size=50)
        assert s.llm_provider == "anthropic"
        assert s.llm_model == "claude-3"
        assert s.session_size == 50


class TestLoadSaveSettings:
    def test_load_from_file(self, tmp_path):
        config = {"llm_provider": "openai", "session_size": 30}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with patch("vocab_trainer.config.CONFIG_PATH", config_path):
            s = load_settings()
        assert s.llm_provider == "openai"
        assert s.session_size == 30
        # Defaults for unspecified fields
        assert s.tts_provider == "edge-tts"

    def test_load_missing_file(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        with patch("vocab_trainer.config.CONFIG_PATH", config_path):
            s = load_settings()
        assert s.llm_provider == "ollama"  # all defaults

    def test_save_creates_file(self, tmp_path):
        config_path = tmp_path / "config.json"
        with patch("vocab_trainer.config.CONFIG_PATH", config_path):
            s = Settings(llm_provider="anthropic")
            save_settings(s)

        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["llm_provider"] == "anthropic"

    def test_unknown_keys_ignored(self, tmp_path):
        config = {"llm_provider": "ollama", "unknown_key": "value"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with patch("vocab_trainer.config.CONFIG_PATH", config_path):
            s = load_settings()
        assert s.llm_provider == "ollama"
        assert not hasattr(s, "unknown_key")
