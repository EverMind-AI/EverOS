"""Contract tests for ``integrations/hermes/cli.py`` (the ``hermes everos`` CLI).

Pins the Hermes-side CLI surface discovered by Hermes' plugin loader:

- ``register_cli`` builds the four subcommands (status/search/flush/setup)
  with the expected argument routing.
- ``_active_scope`` selects ``user_id`` vs ``agent_id`` from ``owner``.
- ``_load_config`` / ``_save_config`` read/write ``$HERMES_HOME/everos.json``
  (0o600 atomic on write).
- ``_redact`` masks any ``*_key`` / ``*_token`` field so ``hermes everos
  setup`` never echoes a secret to stdout.
- ``_cmd_search`` / ``_cmd_flush`` happy paths route the right body to the
  EverOS HTTP client (faked via ``httpx.MockTransport``).
- ``_breaker_state`` returns ``None`` when no provider is loaded.

Hermes-only symbols (``agent``, ``hermes_constants``, ``tools``, ``utils``)
are injected via ``sys.modules`` (see ``tests.helpers.hermes_stub``) so the
plugin bundle imports cleanly without a Hermes runtime. No real network.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import stat
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module", autouse=True)
def _hermes_stubs():
    """Inject Hermes stubs into ``sys.modules`` and import the plugin CLI.

    Mirrors ``test_provider.py``: points ``agent``/``hermes_constants``/
    ``tools``/``utils`` at ``tests.helpers.hermes_stub`` and adds the repo
    root to ``sys.path`` so ``integrations.hermes.cli`` imports as a real
    package.
    """
    import tests.helpers.hermes_stub as stub

    mods = {
        "agent": types.ModuleType("agent"),
        "agent.memory_provider": stub,
        "hermes_constants": stub,
        "tools": types.ModuleType("tools"),
        "tools.registry": stub,
        "utils": stub,
    }
    mods["agent"].memory_provider = stub  # type: ignore[attr-defined]
    mods["tools"].registry = stub  # type: ignore[attr-defined]
    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        import importlib

        cli = importlib.import_module("integrations.hermes.cli")
        yield cli
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(_REPO_ROOT))


@pytest.fixture
def cli(_hermes_stubs):
    return _hermes_stubs


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway ``$HERMES_HOME`` so config reads/writes are hermetic."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("EVEROS_API_KEY", raising=False)
    return home


# ── argparse registration ───────────────────────────────────────────────────


def _build_parser(cli) -> argparse.ArgumentParser:
    """Build a top-level parser with the ``everos`` subcommand wired up."""
    parser = argparse.ArgumentParser(prog="hermes")
    subs = parser.add_subparsers(dest="cmd")
    everos_p = subs.add_parser("everos")
    cli.register_cli(everos_p)
    return parser


def test_register_cli_builds_four_subcommands(cli):
    parser = _build_parser(cli)
    # Each subcommand parses cleanly and sets the everos_action dest. ``search``
    # takes a positional query; the others take only optional flags.
    argv_for = {
        "status": ["everos", "status"],
        "search": ["everos", "search", "q"],
        "flush": ["everos", "flush"],
        "setup": ["everos", "setup"],
    }
    for sub, argv in argv_for.items():
        assert parser.parse_args(argv).everos_action == sub
    # An unknown subcommand is rejected by argparse.
    with pytest.raises(SystemExit):
        parser.parse_args(["everos", "bogus"])


def test_register_cli_search_defaults_and_choices(cli):
    parser = _build_parser(cli)
    args = parser.parse_args(["everos", "search", "hello"])
    assert args.everos_action == "search"
    assert args.query == "hello"
    assert args.owner == "user"
    assert args.method == "hybrid"
    assert args.top_k == 5
    assert args.func is cli.everos_command


def test_register_cli_setup_accepts_all_flags(cli):
    parser = _build_parser(cli)
    args = parser.parse_args(
        [
            "everos",
            "setup",
            "--mode",
            "oss",
            "--api-url",
            "http://x",
            "--api-key",
            "sk-secret",
            "--user-id",
            "alice",
            "--agent-id",
            "bot",
            "--app-id",
            "app",
            "--project-id",
            "proj",
        ]
    )
    assert args.mode == "oss"
    assert args.api_url == "http://x"
    assert args.api_key == "sk-secret"
    assert args.user_id == "alice"


def test_register_cli_flush_defaults(cli):
    parser = _build_parser(cli)
    args = parser.parse_args(["everos", "flush"])
    assert args.everos_action == "flush"
    assert args.session_id == ""
    assert args.owner == "user"


# ── _active_scope ───────────────────────────────────────────────────────────


def test_active_scope_user_selects_user_id(cli):
    cfg = {
        "mode": "platform",
        "user_id": "alice",
        "agent_id": "bot",
        "app_id": "app",
        "project_id": "proj",
    }
    scope = cli._active_scope(cfg, owner="user")
    assert scope["user_id"] == "alice"
    assert "agent_id" not in scope
    assert scope["app_id"] == "app"
    assert scope["project_id"] == "proj"
    assert scope["mode"] == "platform"


def test_active_scope_agent_selects_agent_id(cli):
    cfg = {
        "mode": "oss",
        "user_id": "alice",
        "agent_id": "bot",
        "app_id": "app",
        "project_id": "proj",
    }
    scope = cli._active_scope(cfg, owner="agent")
    assert scope["agent_id"] == "bot"
    assert "user_id" not in scope
    assert scope["mode"] == "oss"


def test_active_scope_defaults_when_keys_missing(cli):
    scope = cli._active_scope({}, owner="user")
    assert scope["user_id"] == "hermes-user"
    assert scope["app_id"] == "default"
    assert scope["project_id"] == "default"


# ── _load_config / _save_config ─────────────────────────────────────────────


def test_load_config_defaults_when_no_file(cli, hermes_home: Path):
    cfg = cli._load_config()
    assert cfg["mode"] == "platform"
    assert cfg["api_url"] == "http://127.0.0.1:8000"
    assert cfg["user_id"] == "hermes-user"
    assert cfg["api_key"] == ""
    assert not (hermes_home / "everos.json").exists()


def test_load_config_reads_hermes_home_json(cli, hermes_home: Path):
    (hermes_home / "everos.json").write_text(
        json.dumps({"mode": "oss", "user_id": "alice", "api_key": "sk-x"})
    )
    cfg = cli._load_config()
    assert cfg["mode"] == "oss"
    assert cfg["user_id"] == "alice"
    assert cfg["api_key"] == "sk-x"
    # Defaults still present for unspecified keys.
    assert cfg["app_id"] == "default"


def test_save_config_writes_mode_0600(cli, hermes_home: Path):
    path = cli._save_config({"api_url": "http://new", "mode": "oss"})
    assert path == hermes_home / "everos.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_url"] == "http://new"
    assert data["mode"] == "oss"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_save_config_merges_existing(cli, hermes_home: Path):
    cli._save_config({"api_url": "http://a", "user_id": "alice"})
    cli._save_config({"mode": "oss"})
    data = json.loads((hermes_home / "everos.json").read_text(encoding="utf-8"))
    assert data["api_url"] == "http://a"
    assert data["user_id"] == "alice"
    assert data["mode"] == "oss"


# ── _redact ─────────────────────────────────────────────────────────────────


def test_redact_masks_api_key(cli):
    out = cli._redact({"api_key": "sk-secret", "mode": "platform"})
    assert out["api_key"] == "***"
    assert out["mode"] == "platform"


def test_redact_masks_key_and_token_suffixes(cli):
    out = cli._redact(
        {"refresh_token": "rt", "service_key": "sk", "user_id": "alice", "mode": "oss"}
    )
    assert out["refresh_token"] == "***"
    assert out["service_key"] == "***"
    assert out["user_id"] == "alice"
    assert out["mode"] == "oss"


def test_redact_does_not_mutate_input(cli):
    values = {"api_key": "sk-secret", "mode": "platform"}
    cli._redact(values)
    assert values["api_key"] == "sk-secret"


# ── _cmd_search / _cmd_flush happy paths ────────────────────────────────────


def _mock_client_factory(handler):
    """Return a callable that builds an httpx.Client backed by MockTransport."""
    return lambda cfg: httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


def test_cmd_search_routes_body_and_prints_json(
    cli,
    hermes_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"episodes": []}})

    monkeypatch.setattr(cli, "_client", _mock_client_factory(handler))

    parser = _build_parser(cli)
    args = parser.parse_args(["everos", "search", "tea", "--owner", "agent"])
    assert cli.everos_command(args) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["data"]["episodes"] == []
    assert captured["url"].endswith("/api/v1/memory/search")
    assert captured["body"]["query"] == "tea"
    assert captured["body"]["agent_id"] == "hermes"
    assert "user_id" not in captured["body"]


def test_cmd_flush_routes_session_and_scope(
    cli,
    hermes_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"status": "extracted"}})

    monkeypatch.setattr(cli, "_client", _mock_client_factory(handler))

    parser = _build_parser(cli)
    args = parser.parse_args(
        ["everos", "flush", "--session-id", "sess-42", "--owner", "user"]
    )
    assert cli.everos_command(args) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["data"]["status"] == "extracted"
    assert captured["body"]["session_id"] == "sess-42"


def test_cmd_setup_redacts_api_key_in_stdout(
    cli, hermes_home: Path, capsys: pytest.CaptureFixture[str]
):
    parser = _build_parser(cli)
    args = parser.parse_args(
        ["everos", "setup", "--api-key", "sk-topsecret", "--mode", "oss"]
    )
    assert cli.everos_command(args) == 0
    out = capsys.readouterr().out
    assert "sk-topsecret" not in out
    assert "***" in out
    # The written file retains the real key (redaction is echo-only).
    data = json.loads((hermes_home / "everos.json").read_text(encoding="utf-8"))
    assert data["api_key"] == "sk-topsecret"


# ── _breaker_state ──────────────────────────────────────────────────────────


def test_breaker_state_returns_none_when_no_provider(cli):
    # The hermes_stub does not expose an active provider, so the best-effort
    # read must degrade to None rather than raising.
    assert cli._breaker_state() is None
