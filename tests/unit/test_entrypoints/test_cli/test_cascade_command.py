"""``everos cascade`` — structural smoke + pure helper tests.

The orchestrator paths require live sqlite + lancedb singletons; those
are exercised by integration tests. Here we cover:

- subcommand registration (sync / status / fix)
- ``--help`` exit codes
- ``_resolve_relative`` (path arithmetic vs. memory root)
- ``_print_failed_table`` (formatting of failed rows)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import cascade as cascade_mod


def test_app_registers_three_commands() -> None:
    names = {cmd.name for cmd in cascade_mod.app.registered_commands}
    assert names == {"sync", "status", "fix"}


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "status" in result.stdout
    assert "fix" in result.stdout


def test_resolve_relative_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    from everos.config import load_settings

    load_settings.cache_clear()

    rel = cascade_mod._resolve_relative(tmp_path / "users" / "u1" / "x.md")
    assert rel == "users/u1/x.md"


def test_resolve_relative_outside_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path / "memory"))
    from everos.config import load_settings

    load_settings.cache_clear()

    other = tmp_path / "somewhere-else.md"
    with pytest.raises(typer.BadParameter, match="not under memory root"):
        cascade_mod._resolve_relative(other)


@dataclass
class _FailedRow:
    md_path: str
    retryable: bool
    retry_count: int
    last_attempt_at: object
    error: str | None


def test_print_failed_table_formats_rows(capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    rows = [
        _FailedRow(
            md_path="users/u1/a.md",
            retryable=True,
            retry_count=2,
            last_attempt_at=datetime(2026, 1, 1, tzinfo=UTC),
            error="boom",
        ),
        _FailedRow(
            md_path="users/u2/b.md",
            retryable=False,
            retry_count=5,
            last_attempt_at=None,
            error=None,
        ),
    ]
    cascade_mod._print_failed_table(rows)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "2 failed row(s):" in out
    assert "users/u1/a.md" in out
    assert "TRUE" in out
    assert "users/u2/b.md" in out
    assert "FALSE" in out
    # Header row present
    assert "md_path" in out and "retries" in out
