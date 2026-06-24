"""CLI root app — verifies sub-typer wiring + ``--help`` exit code."""

from __future__ import annotations

from typer.testing import CliRunner

from everos.entrypoints.cli.main import app


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "everos" in result.stdout
    assert "server" in result.stdout
    assert "cascade" in result.stdout
    assert "demo" in result.stdout
    assert "config" in result.stdout


def test_no_args_shows_help_and_exits_nonzero() -> None:
    # ``no_args_is_help=True`` triggers a help exit with code 2 (typer default).
    result = CliRunner().invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.stdout or "Usage" in result.stderr
