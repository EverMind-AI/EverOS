"""Preflight short-circuits of the three backfill phases, and their exit codes.

Two friendly-error contracts:

1. Missing capability -- without an embedding provider every phase stops
   BEFORE its scan and before the y/N prompt, with a toml hint, and
   ``run_backfill`` maps that to exit ``2`` (``FAILED``) rather than ``1``
   ("aborted by user") or the generic "see logs" bucket.
2. Server running -- when another process holds the memory root's lock,
   every phase stops before it starts an engine or spends a token and
   before the prompt, and ``run_backfill`` maps that to exit ``3``
   (``SERVER_RUNNING``) without a traceback. A server that takes the lock
   between the probe and the replay engine's start is reported the same.

Order inside each phase is capability, then lock: a permanent gap wins over
a transient one. Ctrl-C at the prompt (typer's or click's ``Abort``) is an
interrupt: exit ``130`` with the resume hint.
"""

from __future__ import annotations

from collections.abc import Callable

import click
import pytest
import typer

from everos.core.errors import MemoryRootLockedError
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.service import (
    BackfillDeps,
    BackfillOutcome,
    BackfillSummary,
    NullBackfillPresenter,
    OpsService,
)
from tests.fakes import (
    FakeEmbeddingProvider,
    FakeOmeLock,
    InMemoryClusterStore,
    InMemoryVectorTable,
    ScriptedReplayEngine,
    backfill_deps,
    episode_row,
    make_ops,
)


class _FailingConfirmPresenter(NullBackfillPresenter):
    """Asserts control never reaches the y/N prompt."""

    async def confirm(self, prompt: str, *, auto_yes: bool) -> bool:
        raise AssertionError(
            "presenter.confirm called after a preflight blocker — the "
            "preflight must short-circuit before the y/N prompt"
        )


class _LockMustNotBeProbed(FakeOmeLock):
    def is_free(self) -> bool:
        raise AssertionError(
            "the lock was probed after the capability preflight failed — "
            "capability is a permanent gap; probing the transient lock adds "
            "nothing"
        )


class _UnreadTable(InMemoryVectorTable):
    async def scan(self, where=None):  # type: ignore[no-untyped-def]
        raise AssertionError(
            f"{self.table_name} was scanned before the preflight — a large "
            "backlog would pay an O(N) scan the user cannot act on"
        )


def _with_work() -> dict[str, InMemoryVectorTable]:
    """One episode without vectors: enough for every phase to have work."""
    return {"episode": InMemoryVectorTable("episode", [episode_row()])}


def _unreadable() -> dict[str, InMemoryVectorTable]:
    return {name: _UnreadTable(name) for name in ("episode", "agent_case")}


def _blocked_by_server(**overrides: object) -> BackfillDeps:
    return backfill_deps(
        embedding=FakeEmbeddingProvider(),
        ome_lock=FakeOmeLock(free=False),
        tables=_unreadable(),
        **overrides,
    )


def _no_embedding(**overrides: object) -> BackfillDeps:
    return backfill_deps(
        embedding=None,
        ome_lock=_LockMustNotBeProbed(),
        tables=_unreadable(),
        **overrides,
    )


async def _run(
    deps: BackfillDeps, phase: str
) -> tuple[BackfillOutcome, BackfillSummary]:
    summary = BackfillSummary()
    outcome = await make_ops(backfill=deps).backfill(
        phase, auto_yes=True, presenter=_FailingConfirmPresenter(), summary=summary
    )
    return outcome, summary


# ── server running ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("phase", ["vectors", "clusters", "skills"])
async def test_a_running_server_blocks_every_phase_before_any_work(
    phase: str,
) -> None:
    """Regression (PR #361 J10): the vectors phase used to preflight only the
    capability, so ``--phase all`` against a live server spent its whole
    embedding budget before the clusters phase stopped."""
    outcome, summary = await _run(_blocked_by_server(), phase)

    assert outcome is BackfillOutcome.BLOCKED_BY_SERVER
    result = getattr(summary, phase)
    assert result.aborted is True
    assert result.blocked_by_server is True


@pytest.mark.parametrize("phase", ["vectors", "clusters", "skills"])
async def test_run_backfill_exits_3_when_a_server_runs(
    phase: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = await run_backfill(
        make_ops(backfill=_blocked_by_server()), phase=phase, auto_yes=True
    )
    assert exit_code == 3

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Stop your everos server first" in combined
    assert "SERVER_RUNNING" in combined
    # The exception class by name, not a generic "Traceback": unrelated
    # tests occasionally emit a stdlib "Logging error" traceback.
    assert "EngineLockHeldError" not in combined
    assert "MemoryRootLockedError" not in combined


async def test_a_server_starting_after_the_probe_still_reports_blocked() -> None:
    """The probe found the lock free, but a server took it before the replay
    engine started: the phase reports blocked, it does not leak the error."""
    engine = ScriptedReplayEngine(start_error=MemoryRootLockedError("race"))
    deps = backfill_deps(
        embedding=FakeEmbeddingProvider(),
        tables=_with_work(),
        clusters=InMemoryClusterStore(),
        engine_for=engine.build,
    )
    summary = BackfillSummary()

    outcome = await make_ops(backfill=deps).backfill(
        "clusters", auto_yes=True, presenter=NullBackfillPresenter(), summary=summary
    )

    assert outcome is BackfillOutcome.BLOCKED_BY_SERVER
    assert summary.clusters is not None
    assert summary.clusters.aborted is True
    assert engine.emitted == []


# ── missing capability ───────────────────────────────────────────────────────


@pytest.mark.parametrize("phase", ["vectors", "clusters", "skills"])
async def test_missing_embedding_blocks_every_phase_before_scan_and_lock(
    phase: str,
) -> None:
    """Round-2 finding #5: the capability check runs before the scan, so a
    500k-row backlog without embedding exits at once with the toml hint."""
    outcome, summary = await _run(_no_embedding(), phase)

    assert outcome is BackfillOutcome.BLOCKED_BY_CAPABILITY
    result = getattr(summary, phase)
    assert result.blocked_by_capability == "embedding"
    assert result.blocked_by_server is False
    assert result.aborted is False


async def test_an_empty_store_without_embedding_still_gets_the_hint() -> None:
    """A fresh install sees the toml hint, not "nothing to backfill"."""
    messages: list[str] = []

    class _Recording(_FailingConfirmPresenter):
        def nothing_to_backfill(
            self, message: str, *, scan_failed: bool = False
        ) -> None:
            messages.append(message)

    summary = BackfillSummary()
    outcome = await make_ops(backfill=backfill_deps(embedding=None)).backfill(
        "vectors", auto_yes=True, presenter=_Recording(), summary=summary
    )

    assert outcome is BackfillOutcome.BLOCKED_BY_CAPABILITY
    assert messages == []


async def test_run_backfill_exits_2_with_the_toml_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = await run_backfill(
        make_ops(backfill=backfill_deps(embedding=None, tables=_with_work())),
        phase="vectors",
        auto_yes=True,
    )
    assert exit_code == 2

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "everos.toml" in combined
    assert "[embedding]" in combined
    assert "FAILED" in combined
    # The phase's own hint must not be swallowed by the generic error path.
    assert "Backfill failed — see logs for details." not in combined


# ── Ctrl-C at the prompt ─────────────────────────────────────────────────────


def _abort_with(error: type[BaseException]) -> Callable[..., bool]:
    def _confirm(*_args: object, **_kwargs: object) -> bool:
        raise error()

    return _confirm


@pytest.mark.parametrize(
    "abort",
    [click.exceptions.Abort, typer.Abort],
    ids=["click-abort", "typer-abort"],
)
async def test_ctrl_c_at_the_prompt_exits_130(
    abort: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``typer.confirm`` turns Ctrl-C into an ``Abort`` (a RuntimeError, not a
    KeyboardInterrupt). Typer vendors click, so ``typer.Abort`` and
    ``click.exceptions.Abort`` are distinct classes and both must map to the
    interrupt exit, not fall through to the generic exit 2."""
    assert typer.Abort is not click.exceptions.Abort
    monkeypatch.setattr(typer, "confirm", _abort_with(abort))
    ops: OpsService = make_ops(
        backfill=backfill_deps(embedding=FakeEmbeddingProvider(), tables=_with_work())
    )

    exit_code = await run_backfill(ops, phase="vectors", auto_yes=False)

    assert exit_code == 130
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Interrupted — partial progress was written." in combined
    assert "INTERRUPTED" in combined
    assert "Backfill failed — see logs for details." not in combined
