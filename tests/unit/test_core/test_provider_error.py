"""Unit tests for ProviderNotConfiguredError."""

from __future__ import annotations

from everos.core.errors import ProviderNotConfiguredError


def test_basic_error_has_provider_field():
    exc = ProviderNotConfiguredError(provider="embedding")
    assert exc.provider == "embedding"


def test_error_message_names_provider():
    exc = ProviderNotConfiguredError(provider="embedding")
    assert "'embedding'" in str(exc)


def test_error_message_includes_feature_when_given():
    exc = ProviderNotConfiguredError(provider="rerank", feature="knowledge")
    assert "required by knowledge" in str(exc)


def test_error_message_omits_feature_when_absent():
    exc = ProviderNotConfiguredError(provider="rerank")
    assert "required by" not in str(exc)


def test_error_message_includes_toml_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    exc = ProviderNotConfiguredError(provider="embedding")
    assert str(tmp_path) in str(exc)
    assert "[embedding]" in str(exc)


def test_alternative_hint_appended_when_given():
    exc = ProviderNotConfiguredError(
        provider="rerank",
        feature="agent_hybrid",
        alternative_hint="Set enable_llm_rerank=true to use LLM lane.",
    )
    assert "Alternative: Set enable_llm_rerank=true" in str(exc)


def test_no_env_var_in_message():
    exc = ProviderNotConfiguredError(provider="llm")
    assert "EVEROS_" not in str(exc)


def test_unknown_provider_name_appears_in_section_bracket():
    """Unknown provider names appear as-is in the error message's [section] bracket."""
    exc = ProviderNotConfiguredError(provider="future_provider_xyz")
    assert "future_provider_xyz" in str(exc)
    assert "[future_provider_xyz]" in str(exc)
