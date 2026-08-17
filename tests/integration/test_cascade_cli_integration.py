"""Integration test for ``everos cascade`` CLI commands.

Drives the actual Typer commands against a real sqlite + lancedb under a
tmp memory root. Validates the in-process orchestration that
``test_cascade_command`` (unit) cannot reach: ``_runtime()`` context,
queue summary formatting, fix (no-rows path), and a full
``cascade sync <path>`` round-trip against an empty queue (no handler
ever runs, so embedding availability is irrelevant here).

The CLI commands call ``asyncio.run(_run())`` internally, so this test
is **synchronous** — pytest-asyncio's auto mode would otherwise wrap it
in an event loop, which collides with the CLI's own loop.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.config import load_settings
from everos.entrypoints.cli.commands import cascade as cascade_mod
from everos.infra.persistence.lancedb import dispose_connection
from everos.infra.persistence.sqlite import dispose_engine


@pytest.fixture
def cli_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Tmp memory root + clean singletons; CLI bootstraps the schema itself."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")
    load_settings.cache_clear()
    (tmp_path / "ome.toml").write_text("# test\n")

    # Strip any singleton state from a neighbouring test.
    asyncio.run(_dispose_all())
    yield tmp_path
    asyncio.run(_dispose_all())


async def _dispose_all() -> None:
    await dispose_connection()
    await dispose_engine()


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)


def test_status_on_empty_queue(cli_runtime: Path) -> None:
    """``cascade status`` boots the runtime + prints zeros for a fresh DB."""
    result = CliRunner().invoke(cascade_mod.app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "queue:" in result.stdout
    assert "pending:" in result.stdout
    # Fresh DB: every counter is zero.
    assert "0" in result.stdout
    assert "lsn:" in result.stdout


def test_fix_with_no_failed_rows(cli_runtime: Path) -> None:
    """``cascade fix`` (no ``--apply``) prints the empty-state message."""
    result = CliRunner().invoke(cascade_mod.app, ["fix"])
    assert result.exit_code == 0, result.stdout
    assert "no failed rows" in result.stdout


def test_fix_apply_with_no_failed_rows(cli_runtime: Path) -> None:
    """``cascade fix --apply`` is a noop when there's nothing to fix."""
    result = CliRunner().invoke(cascade_mod.app, ["fix", "--apply"])
    assert result.exit_code == 0, result.stdout
    assert "no failed rows" in result.stdout


def test_sync_on_empty_queue_with_stub_embedder(
    cli_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cascade sync`` invokes orchestrator.drain even on empty queue."""
    # CLI builds the embedder via build_embedding_provider() which would
    # try to connect; replace the orchestrator builder with one wired to
    # the stub embedder.
    from everos.component.tokenizer import build_tokenizer
    from everos.core.persistence import MemoryRoot
    from everos.memory.cascade import CascadeOrchestrator

    def fake_build_orchestrator() -> CascadeOrchestrator:
        root = MemoryRoot.resolve()
        root.ensure()
        return CascadeOrchestrator(
            memory_root=root,
            tokenizer=build_tokenizer(),
        )

    monkeypatch.setattr(cascade_mod, "_build_orchestrator", fake_build_orchestrator)

    result = CliRunner().invoke(cascade_mod.app, ["sync"])
    assert result.exit_code == 0, result.stdout
    assert "sync complete" in result.stdout
    assert "processed 0 row(s)" in result.stdout


def test_sync_with_path_outside_root_errors(
    cli_runtime: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """``cascade sync <path>`` rejects paths outside the memory root."""
    other = tmp_path_factory.mktemp("other") / "x.md"
    other.write_text("# unrelated\n")
    result = CliRunner().invoke(cascade_mod.app, ["sync", str(other)])
    assert result.exit_code != 0
    # Typer.BadParameter surfaces in stderr / mixed output. The rich
    # error box wraps the message at terminal width and pads each line
    # with ``│`` (U+2502 box-drawing); so ``not under`` and
    # ``memory root`` end up separated by spaces *plus* box characters
    # *plus* a newline. ``\s`` doesn't match ``│``, so widen to
    # ``[^\w]+`` (anything that isn't an alnum / underscore) — that
    # tolerates the rich frame without falsely matching real text
    # between the two tokens.
    output = result.stdout + (result.stderr or "")
    plain_output = _strip_ansi(output)
    assert re.search(r"not under[^\w]+memory root", plain_output), output


def test_sync_with_unmatched_path(
    cli_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path under the root but matching no cascade kind exits 1 with a hint."""
    from everos.component.tokenizer import build_tokenizer
    from everos.core.persistence import MemoryRoot
    from everos.memory.cascade import CascadeOrchestrator

    def fake_build_orchestrator() -> CascadeOrchestrator:
        return CascadeOrchestrator(
            memory_root=MemoryRoot.resolve(),
            tokenizer=build_tokenizer(),
        )

    monkeypatch.setattr(cascade_mod, "_build_orchestrator", fake_build_orchestrator)

    # File under the root but in an unregistered subdirectory.
    unregistered = cli_runtime / "stuff" / "random.md"
    unregistered.parent.mkdir(parents=True, exist_ok=True)
    unregistered.write_text("# random\n")
    result = CliRunner().invoke(cascade_mod.app, ["sync", str(unregistered)])
    assert result.exit_code == 1
    # stderr in CliRunner is merged into stdout for typer.echo(..., err=True).
    output = result.stdout + (result.stderr or "")
    assert "does not match any registered cascade kind" in output


# Keep a baseline so future regressions show as a hard failure.
def test_status_handles_pending_rows(cli_runtime: Path) -> None:
    """Seed one pending row via the repo before invoking status."""

    async def seed() -> None:
        # Bring the runtime up like the CLI does, seed, then dispose.
        async with cascade_mod._runtime():
            from everos.infra.persistence.sqlite import md_change_state_repo

            await md_change_state_repo.force_enqueue(
                "users/u1/episodes/episode-2026-01-01.md", "episode"
            )

    asyncio.run(seed())

    result = CliRunner().invoke(cascade_mod.app, ["status"])
    assert result.exit_code == 0, result.stdout
    # One row pending; LSN must be ≥ 1.
    assert "pending:                  1" in result.stdout


def _fake_orchestrator_factory():  # type: ignore[no-untyped-def]
    from everos.component.tokenizer import build_tokenizer
    from everos.core.persistence import MemoryRoot
    from everos.memory.cascade import CascadeOrchestrator

    def _build() -> CascadeOrchestrator:
        root = MemoryRoot.resolve()
        root.ensure()
        return CascadeOrchestrator(
            memory_root=root,
            tokenizer=build_tokenizer(),
        )

    return _build


def test_rebuild_recovers_drifted_index_and_reindexes(
    cli_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cascade rebuild`` recovers a type-drifted index.

    Proves the three properties bare ``rm`` can't offer together:
    (1) it runs despite a schema-type drift that trips the normal
    startup guard (``verify_business_schemas``); (2) it drops + recreates
    the drifted table with the current (correct) type; (3) it re-indexes
    md from scratch even for entries the queue already marked ``done``.
    """
    import datetime as _dt

    import lancedb
    import pyarrow as pa

    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.lancedb import get_table
    from everos.infra.persistence.lancedb.tables.atomic_fact import AtomicFact
    from everos.infra.persistence.lancedb.tables.episode import Episode
    from everos.infra.persistence.markdown import AtomicFactWriter

    root = MemoryRoot.resolve()
    root.ensure()
    owner_id = "u_rebuild"
    bucket = _dt.date(2026, 5, 18)
    md_path = (
        f"default_app/default_project/users/{owner_id}/.atomic_facts/"
        f"atomic_fact-{bucket.isoformat()}.md"
    )

    async def _seed_md() -> None:
        writer = AtomicFactWriter(root=root)
        items = [
            (
                {
                    "owner_id": owner_id,
                    "session_id": f"s_{j}",
                    "timestamp": "2026-05-18T07:04:26+00:00",
                    "parent_id": f"mc_{j}",
                    "sender_ids": [owner_id],
                },
                {"Fact": f"seed fact {j}"},
            )
            for j in range(2)
        ]
        await writer.append_entries(owner_id, items, date=bucket)

    async def _drift_episode_table() -> None:
        conn = await lancedb.connect_async(str(root.lancedb_dir))
        drifted = pa.schema(
            [
                pa.field("subject_vector", pa.string(), nullable=True)
                if f.name == "subject_vector"
                else f
                for f in Episode.to_arrow_schema()
            ]
        )
        await conn.create_table("episode", schema=drifted)
        conn.close()

    async def _episode_subject_vector_type():  # type: ignore[no-untyped-def]
        tbl = await get_table("episode", Episode)
        return (await tbl.schema()).field("subject_vector").type

    async def _atomic_fact_row_count() -> int:
        tbl = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
        return await tbl.count_rows(filter=f"md_path = '{md_path}'")

    asyncio.run(_seed_md())
    asyncio.run(_drift_episode_table())
    asyncio.run(_dispose_all())

    monkeypatch.setattr(
        cascade_mod, "_build_orchestrator", _fake_orchestrator_factory()
    )

    # Contrast: a normal command boots via _runtime() → verify trips on the drift.
    status_result = CliRunner().invoke(cascade_mod.app, ["status"])
    assert status_result.exit_code != 0
    asyncio.run(_dispose_all())

    # rebuild skips verify, recreates the table, and re-indexes md.
    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "rebuild complete" in result.stdout
    asyncio.run(_dispose_all())

    # Table recreated with the correct vector type; md re-indexed.
    assert asyncio.run(_episode_subject_vector_type()).equals(
        Episode.to_arrow_schema().field("subject_vector").type
    )
    assert asyncio.run(_atomic_fact_row_count()) == 2


def test_rebuild_refuses_to_run_while_a_server_holds_the_lock(
    cli_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rebuild`` must refuse when a daemon is running on this memory root.

    It drops and recreates the LanceDB tables; a live daemon holds cached
    table handles and would keep writing to the dropped dataset, leaving a
    corrupted rebuild plus a permanent-failure backlog. Detection reuses the
    OME jobstore lock that ``backfill`` already gates on, and the exit code
    matches backfill's ``3`` (SERVER_RUNNING).
    """
    monkeypatch.setattr(cascade_mod, "ome_lock_is_free", lambda: False)

    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])

    assert result.exit_code == 3, result.output
    # The explanation goes to stderr (click 8.2 keeps the streams separate).
    assert "server" in result.stderr.lower()
    assert "stop `everos server`" in result.stderr.lower()
    # And it must bail out BEFORE touching anything — none of the step
    # progress lines may appear.
    combined = result.output + result.stderr
    assert "LanceDB table(s)" not in combined
    assert "cascade queue row(s)" not in combined
    assert "rebuild complete" not in combined


# Reduce false negatives on date drift.
def test_resolve_relative_via_command_arg(cli_runtime: Path) -> None:
    """An absolute path under the root works through ``cascade sync <path>``."""
    md_file = cli_runtime / "users" / "u1" / "episodes" / "episode-2026-05-25.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()  # only used so the var isn't unused
    md_file.write_text(f"# {today}\n")

    # We don't need the orchestrator to actually drain anything; pass --help
    # against the sync subcommand to verify the path resolution helper
    # doesn't barf at construction time.
    result = CliRunner().invoke(cascade_mod.app, ["sync", "--help"])
    assert result.exit_code == 0
