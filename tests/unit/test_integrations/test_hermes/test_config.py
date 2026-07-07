"""Contract tests for ``integrations/hermes/_config.py``.

Pins the Hermes-agnostic config layer:

- ``load_config`` merges defaults → env → JSON (JSON wins); a missing or
  malformed JSON file is silently skipped (defaults + env remain).
- ``resolve_user_id`` priority chain: configured (non-default) > kwargs
  ``user_id`` > kwargs ``user_id_alt`` > ``_DEFAULT_USER_ID``; the literal
  default placeholder is treated as unset.
- ``resolve_agent_id`` falls back to ``_DEFAULT_AGENT_ID``.
- ``get_scope_ids`` validates the charset, rejects path-traversal tokens
  (``.`` / ``..``), enforces length 1..128, and defaults empty values.
- ``is_configured`` is true on a non-empty ``api_url`` or a recognised
  ``mode``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hermes._config import (
    get_scope_ids,
    is_configured,
    load_config,
    resolve_agent_id,
    resolve_user_id,
)
from hermes._constants import (
    _DEFAULT_AGENT_ID,
    _DEFAULT_API_URL,
    _DEFAULT_APP_ID,
    _DEFAULT_PROJECT_ID,
    _DEFAULT_USER_ID,
)

# ── load_config ─────────────────────────────────────────────────────────────


def test_load_config_defaults_when_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("EVEROS_API_URL", "EVEROS_USER_ID", "EVEROS_AGENT_ID", "EVEROS_MODE"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(tmp_path)
    assert cfg["api_url"] == _DEFAULT_API_URL
    assert cfg["mode"] == "platform"
    assert cfg["user_id"] == _DEFAULT_USER_ID
    assert cfg["agent_id"] == _DEFAULT_AGENT_ID
    assert cfg["app_id"] == _DEFAULT_APP_ID
    assert cfg["project_id"] == _DEFAULT_PROJECT_ID
    assert cfg["agent_track_enabled"] is False
    assert cfg["everos_root"] is None


def test_load_config_env_overrides_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_API_URL", "http://env.example")
    monkeypatch.setenv("EVEROS_USER_ID", "env-user")
    monkeypatch.setenv("EVEROS_AGENT_ID", "env-agent")
    monkeypatch.setenv("EVEROS_MODE", "oss")
    cfg = load_config(tmp_path)
    assert cfg["api_url"] == "http://env.example"
    assert cfg["user_id"] == "env-user"
    assert cfg["agent_id"] == "env-agent"
    assert cfg["mode"] == "oss"


def test_load_config_json_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_USER_ID", "env-user")
    monkeypatch.setenv("EVEROS_API_URL", "http://env.example")
    (tmp_path / "everos.json").write_text(
        json.dumps(
            {
                "user_id": "json-user",
                "api_url": "http://json.example",
                "mode": "platform",
                "agent_track_enabled": True,
                "everos_root": "/var/everos",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg["user_id"] == "json-user"
    assert cfg["api_url"] == "http://json.example"
    assert cfg["agent_track_enabled"] is True
    assert cfg["everos_root"] == "/var/everos"


def test_load_config_skips_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_USER_ID", "env-user")
    (tmp_path / "everos.json").write_text("{not valid json", encoding="utf-8")
    cfg = load_config(tmp_path)
    # Env value survives; defaults fill the rest.
    assert cfg["user_id"] == "env-user"
    assert cfg["api_url"] == _DEFAULT_API_URL


def test_load_config_skips_empty_json_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_USER_ID", "env-user")
    (tmp_path / "everos.json").write_text(
        json.dumps({"user_id": "", "api_url": None, "mode": "oss"}),
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg["user_id"] == "env-user"
    assert cfg["api_url"] == _DEFAULT_API_URL
    assert cfg["mode"] == "oss"


# ── resolve_user_id ─────────────────────────────────────────────────────────


def test_resolve_user_id_configured_wins() -> None:
    assert resolve_user_id({"user_id": "real-user"}, "kw", "alt") == "real-user"


def test_resolve_user_id_default_placeholder_is_unset() -> None:
    # The literal _DEFAULT_USER_ID is treated as unset → fall through to kwargs.
    assert resolve_user_id({"user_id": _DEFAULT_USER_ID}, "kw-user", "alt") == "kw-user"


def test_resolve_user_id_kwargs_user_then_alt() -> None:
    assert resolve_user_id({}, "kw-user", "alt") == "kw-user"


def test_resolve_user_id_kwargs_alt_when_no_user() -> None:
    assert resolve_user_id({}, None, "alt-user") == "alt-user"


def test_resolve_user_id_falls_back_to_default() -> None:
    assert resolve_user_id({}, None, None) == _DEFAULT_USER_ID


# ── resolve_agent_id ────────────────────────────────────────────────────────


def test_resolve_agent_id_uses_config() -> None:
    assert resolve_agent_id({"agent_id": "real-agent"}) == "real-agent"


def test_resolve_agent_id_default_when_missing() -> None:
    assert resolve_agent_id({}) == _DEFAULT_AGENT_ID


# ── get_scope_ids ───────────────────────────────────────────────────────────


def test_get_scope_ids_valid() -> None:
    ids = get_scope_ids({"app_id": "my.app-1", "project_id": "proj_2"})
    assert ids.app_id == "my.app-1"
    assert ids.project_id == "proj_2"


def test_get_scope_ids_defaults_when_empty() -> None:
    ids = get_scope_ids({})
    assert ids.app_id == _DEFAULT_APP_ID
    assert ids.project_id == _DEFAULT_PROJECT_ID


def test_get_scope_ids_none_values_default() -> None:
    ids = get_scope_ids({"app_id": None, "project_id": None})
    assert ids.app_id == _DEFAULT_APP_ID
    assert ids.project_id == _DEFAULT_PROJECT_ID


def test_get_scope_ids_rejects_bad_charset() -> None:
    with pytest.raises(ValueError, match="charset"):
        get_scope_ids({"app_id": "a/b"})


def test_get_scope_ids_rejects_traversal_dot() -> None:
    with pytest.raises(ValueError, match="traversal"):
        get_scope_ids({"app_id": "."})


def test_get_scope_ids_rejects_traversal_double_dot() -> None:
    with pytest.raises(ValueError, match="traversal"):
        get_scope_ids({"project_id": ".."})


def test_get_scope_ids_rejects_too_long() -> None:
    too_long = "a" * 129
    with pytest.raises(ValueError, match="length"):
        get_scope_ids({"app_id": too_long})


def test_get_scope_ids_accepts_max_length() -> None:
    max_len = "a" * 128
    ids = get_scope_ids({"app_id": max_len})
    assert ids.app_id == max_len


def test_get_scope_ids_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        get_scope_ids({"app_id": 123})


# ── is_configured ───────────────────────────────────────────────────────────


def test_is_configured_true_with_api_url() -> None:
    assert is_configured({"api_url": "http://x", "mode": ""}) is True


def test_is_configured_true_with_recognised_mode() -> None:
    assert is_configured({"api_url": "", "mode": "platform"}) is True
    assert is_configured({"api_url": "", "mode": "oss"}) is True


def test_is_configured_false_with_empty_url_and_unknown_mode() -> None:
    assert is_configured({"api_url": "", "mode": "weird"}) is False


def test_is_configured_false_when_both_missing() -> None:
    assert is_configured({"api_url": "", "mode": ""}) is False
