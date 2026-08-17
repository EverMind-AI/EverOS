"""``everos cascade`` — structural smoke + pure helper tests.

The orchestrator paths require live sqlite + lancedb singletons; those
are exercised by integration tests. Here we cover:

- subcommand registration (sync / status / fix / rebuild)
- ``--help`` exit codes
- ``_resolve_relative`` (path arithmetic vs. memory root)
- ``_print_failed_table`` (formatting of failed rows)
- ``_build_orchestrator`` soft-build (embedding optional)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer
from sqlmodel import SQLModel
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import cascade as cascade_mod
from everos.infra.persistence.sqlite import dispose_engine, get_engine


def test_app_registers_expected_commands() -> None:
    names = {cmd.name for cmd in cascade_mod.app.registered_commands}
    assert names == {"sync", "status", "fix", "backfill", "rebuild"}


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "status" in result.stdout
    assert "fix" in result.stdout
    assert "rebuild" in result.stdout


def test_resolve_relative_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    from everos.config import load_settings

    load_settings.cache_clear()

    rel = cascade_mod._resolve_relative(tmp_path / "users" / "u1" / "x.md")
    assert rel == "users/u1/x.md"


def test_resolve_relative_outside_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path / "memory"))
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


# ── _build_orchestrator soft-build (embedding optional) ─────────────────────


@pytest.fixture(autouse=True)
def _clear_capability_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with a fresh capability lookup."""
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", None)


@pytest.fixture
async def sqlite_schema_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Create the SQLite schema for _build_orchestrator tests."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    await dispose_engine()


async def test_build_orchestrator_without_embedding(
    sqlite_schema_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_build_orchestrator succeeds when embedding is unconfigured (soft-build)."""
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "")  # simulate missing
    from everos.entrypoints.cli.commands.cascade import _build_orchestrator
    from everos.memory.cascade import CascadeOrchestrator

    # Should NOT raise TypeError about missing embedder kwarg
    orchestrator = _build_orchestrator()
    assert orchestrator is not None
    assert isinstance(orchestrator, CascadeOrchestrator)
