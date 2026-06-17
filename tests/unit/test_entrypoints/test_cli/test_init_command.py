"""``everos init`` — CLI behavior + edge cases.

Covers:

- default ``./.env`` path, written with 0600 permissions
- ``--to <path>`` creates parent dirs
- ``--force`` overwrites; without it the command refuses with exit 1
- ``--print`` writes to stdout, NOT to disk
- ``--xdg`` and ``--to`` are mutually exclusive (exit 2)
- ``--xdg`` honors ``XDG_CONFIG_HOME``
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from a fresh tmp cwd so default ``./.env`` lands in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_writes_dotenv_in_cwd(runner: CliRunner, in_tmp: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    written = in_tmp / ".env"
    assert written.exists()
    assert written.stat().st_size > 0
    assert "EVEROS_LLM__API_KEY" in written.read_text()
    assert "https://github.com/EverMind-AI/EverOS/blob/main/QUICKSTART.md" in (
        result.output
    )


def test_default_file_permissions_are_0600(runner: CliRunner, in_tmp: Path) -> None:
    """The generated .env holds API keys — must not be world-readable."""
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    mode = stat.S_IMODE((in_tmp / ".env").stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_refuses_overwrite_without_force(runner: CliRunner, in_tmp: Path) -> None:
    (in_tmp / ".env").write_text("PREEXISTING=1\n")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "already exists" in (result.output + (result.stderr or ""))
    # Original content must be preserved.
    assert (in_tmp / ".env").read_text() == "PREEXISTING=1\n"


def test_force_overwrites(runner: CliRunner, in_tmp: Path) -> None:
    (in_tmp / ".env").write_text("PREEXISTING=1\n")
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0
    body = (in_tmp / ".env").read_text()
    assert "PREEXISTING=1" not in body
    assert "EVEROS_LLM__API_KEY" in body


def test_to_creates_parent_dirs(runner: CliRunner, in_tmp: Path) -> None:
    target = in_tmp / "nested" / "subdir" / ".env"
    result = runner.invoke(app, ["init", "--to", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    assert "EVEROS_LLM__API_KEY" in target.read_text()


def test_print_writes_stdout_not_disk(runner: CliRunner, in_tmp: Path) -> None:
    result = runner.invoke(app, ["init", "--print"])
    assert result.exit_code == 0
    assert "EVEROS_LLM__API_KEY" in result.output
    # No disk side-effect.
    assert not (in_tmp / ".env").exists()


def test_xdg_writes_to_xdg_config_home(
    runner: CliRunner, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg_root = in_tmp / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_root))
    result = runner.invoke(app, ["init", "--xdg"])
    assert result.exit_code == 0
    target = xdg_root / "everos" / ".env"
    assert target.exists()


def test_xdg_falls_back_to_dot_config(
    runner: CliRunner, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``XDG_CONFIG_HOME`` → default ``~/.config``.

    We sandbox ``$HOME`` to ``in_tmp`` so the test does not touch a real
    user's ``~/.config``.
    """
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(in_tmp))
    result = runner.invoke(app, ["init", "--xdg"])
    assert result.exit_code == 0
    target = in_tmp / ".config" / "everos" / ".env"
    assert target.exists()


def test_xdg_and_to_are_mutually_exclusive(runner: CliRunner, in_tmp: Path) -> None:
    result = runner.invoke(app, ["init", "--xdg", "--to", str(in_tmp / "other.env")])
    assert result.exit_code == 2
    assert "mutually exclusive" in (result.output + (result.stderr or ""))


def test_template_resource_is_packaged_under_everos_templates() -> None:
    """The packaged resource must remain at the canonical location.

    Guards the wheel/sdist layout: ``init_cmd`` reads
    ``everos.templates.env.template`` via ``importlib.resources``; if
    someone moves the file without updating ``_TEMPLATE_PACKAGE``, this
    test fails immediately.
    """
    from importlib import resources

    res = resources.files("everos.templates").joinpath("env.template")
    assert res.is_file()
    body = res.read_text(encoding="utf-8")
    assert "EVEROS_LLM__API_KEY" in body
    assert "DASHSCOPE_API_KEY" in body
    assert "text-embedding-v4" in body
    assert "gte-rerank-v2" in body


def test_template_has_single_active_value_per_env_key() -> None:
    """Alternative provider examples must stay commented to avoid overrides."""
    from collections import Counter
    from importlib import resources

    body = (
        resources.files("everos.templates")
        .joinpath("env.template")
        .read_text(encoding="utf-8")
    )
    active_assignments = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    assert all("<" not in line for line in active_assignments)

    keys = [line.split("=", 1)[0] for line in active_assignments]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    assert duplicates == []


# ── 4-layer .env resolution for ``server start`` ────────────────────────


def test_resolve_env_file_explicit_wins(in_tmp: Path) -> None:
    """``--env-file <path>`` beats cwd / XDG / ~/.everos fallbacks."""
    from everos.entrypoints.cli.commands.server import _resolve_env_file

    explicit = in_tmp / "explicit.env"
    explicit.write_text("X=1\n")
    # Also seed cwd .env so we can prove the explicit wins.
    (in_tmp / ".env").write_text("CWD=1\n")
    resolved = _resolve_env_file(str(explicit))
    assert resolved == explicit


def test_resolve_env_file_cwd_wins_over_xdg(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.entrypoints.cli.commands.server import _resolve_env_file

    xdg_root = in_tmp / "xdg"
    (xdg_root / "everos").mkdir(parents=True)
    (xdg_root / "everos" / ".env").write_text("XDG=1\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_root))
    cwd_env = in_tmp / ".env"
    cwd_env.write_text("CWD=1\n")
    resolved = _resolve_env_file(None)
    assert resolved == cwd_env


def test_resolve_env_file_xdg_when_no_cwd(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.entrypoints.cli.commands.server import _resolve_env_file

    xdg_root = in_tmp / "xdg"
    (xdg_root / "everos").mkdir(parents=True)
    target = xdg_root / "everos" / ".env"
    target.write_text("XDG=1\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_root))
    # No cwd/.env.
    resolved = _resolve_env_file(None)
    assert resolved == target


def test_resolve_env_file_everos_home_fallback(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~/.everos/.env`` is the last fallback when nothing else exists."""
    from everos.entrypoints.cli.commands.server import _resolve_env_file

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(in_tmp))
    target = in_tmp / ".everos" / ".env"
    target.parent.mkdir(parents=True)
    target.write_text("EVEROS_ROOT=1\n")
    resolved = _resolve_env_file(None)
    assert resolved == target


def test_resolve_env_file_none_when_no_layer_matches(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four layers absent → ``None`` (the server then falls back to
    inherited process env, which is the documented CI/container path)."""
    from everos.entrypoints.cli.commands.server import _resolve_env_file

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(in_tmp))
    # Nothing in cwd, no XDG path, no ~/.everos/.
    assert not (in_tmp / ".env").exists()
    assert _resolve_env_file(None) is None


# ``os`` imported above just to keep ruff from complaining; remove if Ruff
# F401 hits.
_ = os
