"""Unit tests for ``ObservabilitySettings`` (OpenTelemetry tracing config)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.config import Settings, load_settings
from everos.config.settings import ObservabilitySettings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    load_settings.cache_clear()


def test_observability_defaults_are_off_and_neutral() -> None:
    obs = load_settings().observability
    assert obs.enabled is False
    assert obs.exporter == "otlp_http"
    assert obs.endpoint == ""
    assert obs.headers == {}
    assert obs.service_name == "everos"
    assert obs.sample_rate == 1.0


def test_env_overrides_observability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "r"
    root.mkdir()
    monkeypatch.setenv("EVEROS_OBSERVABILITY__ENABLED", "true")
    monkeypatch.setenv(
        "EVEROS_OBSERVABILITY__ENDPOINT", "https://otlp.example/v1/traces"
    )
    monkeypatch.setenv("EVEROS_OBSERVABILITY__SERVICE_NAME", "everos-test")
    s = Settings(_everos_root=root)
    assert s.observability.enabled is True
    assert s.observability.endpoint == "https://otlp.example/v1/traces"
    assert s.observability.service_name == "everos-test"


def test_sample_rate_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservabilitySettings(sample_rate=1.5)
    with pytest.raises(ValidationError):
        ObservabilitySettings(sample_rate=-0.1)


def test_recall_score_defaults() -> None:
    obs = load_settings().observability
    assert obs.langfuse_public_key is None
    assert obs.langfuse_secret_key is None
    assert obs.langfuse_host is None
    assert obs.emit_recall_scores is True
    assert obs.recall_hit_threshold == 0.6


def test_secret_key_is_not_leaked_in_repr() -> None:
    obs = ObservabilitySettings(langfuse_secret_key="sk-lf-supersecret")
    # SecretStr masks the value in repr/str.
    assert "sk-lf-supersecret" not in repr(obs)
    assert obs.langfuse_secret_key is not None
    assert obs.langfuse_secret_key.get_secret_value() == "sk-lf-supersecret"


def test_env_overrides_recall_score_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "r"
    root.mkdir()
    monkeypatch.setenv("EVEROS_OBSERVABILITY__LANGFUSE_HOST", "https://lf.example")
    monkeypatch.setenv("EVEROS_OBSERVABILITY__EMIT_RECALL_SCORES", "false")
    monkeypatch.setenv("EVEROS_OBSERVABILITY__RECALL_HIT_THRESHOLD", "0.8")
    s = Settings(_everos_root=root)
    assert s.observability.langfuse_host == "https://lf.example"
    assert s.observability.emit_recall_scores is False
    assert s.observability.recall_hit_threshold == 0.8
