"""Contract tests for ``integrations/hermes/_setup.py``.

Pins the Hermes-agnostic setup wizard:

- ``write_everos_toml`` emits a ``tomllib``-parseable file with
  ``[llm]`` / ``[embedding]`` (and ``[rerank]`` when provided) sections
  carrying the caller's keys, chmod 600, and omits ``[rerank]`` when None.
- ``seed_ome_toml`` enables the three agent-track strategies under
  ``[strategies.*]``, returns ``None`` when ``agent_track`` is falsy, and
  merges into an existing ``ome.toml`` without duplicate keys or section
  flattening.
- ``build_everos_json`` returns the expected dict.
- ``post_setup`` in non-interactive mode writes ``everos.json`` (platform)
  or ``everos.toml`` + ``everos.json`` (+ ``ome.toml`` when the agent track
  is on) for oss mode, all parseable.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from hermes._setup import (
    build_everos_json,
    post_setup,
    seed_ome_toml,
    write_everos_toml,
)

_TOML_MODE = 0o600


# ── write_everos_toml ───────────────────────────────────────────────────────


def test_write_everos_toml_sections_keys_and_mode(tmp_path: Path) -> None:
    path = write_everos_toml(
        tmp_path,
        llm={
            "model": "gpt-4",
            "api_key": "secret",
            "base_url": "http://llm",
            "timeout_seconds": 30,
            "max_retries": 2,
        },
        embedding={"model": "text-embed", "base_url": "http://emb"},
        rerank={"model": "rerank-1", "base_url": "http://rr"},
    )
    assert path == tmp_path / "everos.toml"
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == _TOML_MODE
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["llm"]["model"] == "gpt-4"
    assert data["llm"]["api_key"] == "secret"
    assert data["llm"]["base_url"] == "http://llm"
    assert data["llm"]["timeout_seconds"] == 30
    assert data["llm"]["max_retries"] == 2
    assert data["embedding"]["model"] == "text-embed"
    assert data["embedding"]["base_url"] == "http://emb"
    assert data["rerank"]["model"] == "rerank-1"
    assert data["rerank"]["base_url"] == "http://rr"


def test_write_everos_toml_omits_rerank_when_none(tmp_path: Path) -> None:
    path = write_everos_toml(
        tmp_path,
        llm={"model": "m"},
        embedding={"model": "e"},
        rerank=None,
    )
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    assert "llm" in data
    assert "embedding" in data
    assert "rerank" not in data


# ── seed_ome_toml ───────────────────────────────────────────────────────────


def test_seed_ome_toml_writes_three_strategies(tmp_path: Path) -> None:
    path = seed_ome_toml(tmp_path, agent_track=True)
    assert path is not None
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == _TOML_MODE
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    strategies = data["strategies"]
    for name in (
        "extract_agent_case",
        "extract_agent_skill",
        "trigger_skill_clustering",
    ):
        assert strategies[name]["enabled"] is True


def test_seed_ome_toml_returns_none_when_disabled(tmp_path: Path) -> None:
    assert seed_ome_toml(tmp_path, agent_track=False) is None
    assert not (tmp_path / "ome.toml").exists()


def test_seed_ome_toml_merges_without_duplicates_or_flattening(
    tmp_path: Path,
) -> None:
    existing = (
        "# pre-existing\n\n[strategies.foo]\nenabled = false\n\n[other]\nkey = 1\n"
    )
    (tmp_path / "ome.toml").write_text(existing, encoding="utf-8")
    path = seed_ome_toml(tmp_path, agent_track=True)
    assert path is not None
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    # Pre-existing keys are preserved.
    assert data["strategies"]["foo"]["enabled"] is False
    assert data["other"]["key"] == 1
    # The three agent-track strategies are enabled.
    for name in (
        "extract_agent_case",
        "extract_agent_skill",
        "trigger_skill_clustering",
    ):
        assert data["strategies"][name]["enabled"] is True
    # No flattening: strategies is a nested table, not top-level scalar keys.
    assert "extract_agent_case" in data["strategies"]
    assert "extract_agent_case" not in data


# ── build_everos_json ───────────────────────────────────────────────────────


def test_build_everos_json_returns_expected_dict() -> None:
    payload = build_everos_json(
        api_url="http://api",
        mode="oss",
        user_id="u-1",
        agent_id="a-1",
        agent_track_enabled=True,
        everos_root="/root",
    )
    assert payload == {
        "api_url": "http://api",
        "mode": "oss",
        "user_id": "u-1",
        "agent_id": "a-1",
        "agent_track_enabled": True,
        "everos_root": "/root",
    }


def test_build_everos_json_normalises_agent_track_bool() -> None:
    payload = build_everos_json(
        api_url="",
        mode="platform",
        user_id="u",
        agent_id="a",
        agent_track_enabled=1,  # truthy non-bool
        everos_root=None,
    )
    assert payload["agent_track_enabled"] is True


# ── post_setup ──────────────────────────────────────────────────────────────


def test_post_setup_platform_writes_only_everos_json(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes"
    inputs = {
        "mode": "platform",
        "api_url": "http://api",
        "user_id": "u",
        "agent_id": "a",
        "agent_track_enabled": False,
    }
    payload = post_setup(hermes_home, {}, interactive=False, inputs=inputs)
    json_path = hermes_home / "everos.json"
    assert json_path.exists()
    assert (json_path.stat().st_mode & 0o777) == _TOML_MODE
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert payload["mode"] == "platform"
    assert payload["everos_root"] is None
    # Platform mode does not write everos.toml under hermes_home.
    assert not (hermes_home / "everos.toml").exists()


def test_post_setup_oss_writes_toml_json_and_ome(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes"
    everos_root = tmp_path / "everos"
    inputs = {
        "mode": "oss",
        "everos_root": str(everos_root),
        "api_url": "http://api",
        "user_id": "u",
        "agent_id": "a",
        "agent_track_enabled": True,
        "llm": {"model": "m", "api_key": "k", "base_url": "http://llm"},
        "embedding": {"model": "e", "base_url": "http://emb"},
        "rerank": None,
    }
    payload = post_setup(hermes_home, {}, interactive=False, inputs=inputs)

    assert (hermes_home / "everos.json").exists()
    assert (everos_root / "everos.toml").exists()
    assert (everos_root / "ome.toml").exists()  # agent_track on

    with (everos_root / "everos.toml").open("rb") as fh:
        toml_data = tomllib.load(fh)
    assert toml_data["llm"]["model"] == "m"
    assert toml_data["embedding"]["model"] == "e"
    assert "rerank" not in toml_data

    json_data = json.loads((hermes_home / "everos.json").read_text(encoding="utf-8"))
    assert json_data == payload
    assert json_data["mode"] == "oss"
    assert json_data["everos_root"] == str(everos_root)


def test_post_setup_oss_skips_ome_when_agent_track_off(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes"
    everos_root = tmp_path / "everos"
    inputs = {
        "mode": "oss",
        "everos_root": str(everos_root),
        "agent_track_enabled": False,
        "llm": {"model": "m"},
        "embedding": {"model": "e"},
    }
    post_setup(hermes_home, {}, interactive=False, inputs=inputs)
    assert (everos_root / "everos.toml").exists()
    assert not (everos_root / "ome.toml").exists()
    assert (hermes_home / "everos.json").exists()
