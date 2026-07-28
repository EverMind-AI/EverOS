"""Unit tests for missing_config_error helper."""

from __future__ import annotations

from everos.component.utils.config_hints import missing_config_error


def test_error_contains_field_label():
    msg = missing_config_error("LLM api_key", "llm")
    assert "LLM api_key is not configured" in msg


def test_error_contains_actual_root_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    msg = missing_config_error("LLM api_key", "llm")
    assert str(tmp_path) in msg
    assert "everos.toml" in msg


def test_error_contains_init_hint():
    msg = missing_config_error("LLM api_key", "llm")
    assert "everos init" in msg


def test_error_contains_toml_section():
    msg = missing_config_error("LLM api_key", "llm")
    assert "[llm]" in msg


def test_error_does_not_mention_env_var():
    """Config guidance rule: point at toml only, not env vars."""
    msg = missing_config_error("LLM api_key", "llm")
    assert "EVEROS_" not in msg
    assert "env" not in msg.lower()
