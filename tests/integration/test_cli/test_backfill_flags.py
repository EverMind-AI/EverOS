"""Integration tests for Task 24's ``cascade backfill`` polish pass.

Covers three behaviours layered on top of Task 20-23's scaffold + three
phase implementations:

- **Summary block**: after ``run_backfill`` finishes (or is interrupted),
  a consolidated "Backfill summary" section is printed, one line per
  phase that actually ran, plus a final ``Exit: <LABEL> (<code>)`` line.
- **Ctrl-C handling**: an interrupt raised from inside a phase body must
  propagate up to ``run_backfill``'s own top-level handler (not be
  swallowed mid-phase), which prints a resume hint and returns exit
  code 130 (SIGINT convention: 128 + 2) instead of leaking a bare
  traceback. Covered at two levels: a synchronous ``KeyboardInterrupt``
  raise (cheap unit-level regression guard, but *not* representative of
  a real terminal Ctrl-C — see below), and the actual mechanism a real
  SIGINT uses under ``asyncio.run`` on Python 3.11+ — ``asyncio.Runner``
  translates it into ``main_task.cancel()``, which raises
  ``asyncio.CancelledError`` (a ``BaseException``, not a
  ``KeyboardInterrupt`` subclass) at the currently suspended ``await``.
  A bare ``except KeyboardInterrupt`` alone never sees that exception,
  so the CancelledError path is exercised both by simulation (raising
  ``CancelledError`` from inside an ``await``) and by delivering a real
  ``SIGINT`` to the test process while a mocked phase is parked in
  ``asyncio.sleep``.
- **``--help`` polish**: the CLI help text names every ``--phase`` choice
  and explains what ``--yes`` does.

Phase 1 (``vectors``) is reused for the "real counts" summary case
because it is the cheapest phase to drive end-to-end (mirrors
``backfill_runtime`` in ``test_backfill_phase1.py``); Phases 2/3 are
exercised in the all-zero case only, since Task 24 doesn't add any new
phase behaviour of its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands import cascade as cascade_mod
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import Episode, dispose_connection, episode_repo
from everos.memory.cascade import _backfill as backfill_mod

_DIM = 1024


class _StubEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(len(t) % 7)] * self.dim for t in texts]


@pytest.fixture
async def backfill_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + stub embedder (mirrors ``test_backfill_phase1.py``)."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    await dispose_connection()

    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_StubEmbedder())
    )

    yield tmp_path
    await dispose_connection()


def _episode(entry_id: str) -> Episode:
    return Episode(
        id=f"u1_{entry_id}",
        entry_id=entry_id,
        owner_id="u1",
        owner_type="user",
        timestamp=get_utc_now(),
        parent_id="mc1",
        sender_ids=["u1"],
        episode=f"episode body {entry_id}",
        episode_tokens=f"episode body {entry_id}",
        md_path="users/u1/episodes/episode-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=None,
    )


# ── summary block ────────────────────────────────────────────────────────────


async def test_summary_block_all_zero_on_empty_db(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await run_backfill(phase="all", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Backfill summary" in out
    assert "Phase 1 (vectors)" in out
    assert "0 rows / 0 failed / 0 tokens" in out
    assert "Phase 2 (clusters)" in out
    assert "0 events emitted / 0 clusters created" in out
    assert "Phase 3 (skills)" in out
    assert "0 agent cases processed / 0 skills extracted" in out
    assert "Exit: SUCCESS  (0)" in out


async def test_summary_block_reports_real_counts_for_ran_phase_only(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await episode_repo.add([_episode("ep1"), _episode("ep2")])

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Backfill summary" in out
    assert "Phase 1 (vectors)" in out
    assert "2 rows / 0 failed" in out
    # Only the phase that actually ran gets a summary line.
    assert "Phase 2 (clusters)" not in out
    assert "Phase 3 (skills)" not in out
    assert "Exit: SUCCESS  (0)" in out


# ── Ctrl-C handling ──────────────────────────────────────────────────────────


async def test_synchronous_keyboard_interrupt_mid_phase_returns_130_with_resume_hint(
    backfill_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cheap regression guard for the raw ``except KeyboardInterrupt`` catch.

    This raises synchronously (no ``await`` in between), so the
    exception lands directly in ``run_backfill``'s own frame — it does
    *not* exercise ``asyncio.Runner``'s SIGINT-to-``CancelledError``
    translation that a real terminal Ctrl-C goes through. See
    ``test_cancelled_error_mid_phase_returns_130_with_resume_hint`` and
    ``test_real_sigint_during_phase_await_returns_130_with_resume_hint``
    below for that path.
    """

    async def _raise_interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", _raise_interrupt)

    code = await run_backfill(phase="all", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 130
    assert "Interrupted" in out
    assert "partial progress was written" in out
    assert "Resume by running" in out
    assert "everos cascade backfill --phase <phase-name> --yes" in out
    assert "vectors, clusters, skills, all" in out
    # The summary block still prints on interrupt, reflecting no progress yet.
    assert "Backfill summary" in out
    assert "Exit: INTERRUPTED  (130)" in out


async def test_cancelled_error_mid_phase_returns_130_with_resume_hint(
    backfill_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Simulates what ``asyncio.Runner`` actually does on a real SIGINT.

    On Python 3.11+, ``asyncio.Runner`` (which backs ``asyncio.run``)
    turns a terminal Ctrl-C into ``main_task.cancel()``, which raises
    ``asyncio.CancelledError`` — a ``BaseException``, never a
    ``KeyboardInterrupt`` subclass — at the task's currently suspended
    ``await``. Raising it from inside an ``await`` (rather than
    synchronously) reproduces that mechanism; a bare
    ``except KeyboardInterrupt`` alone would miss it entirely and let it
    escape ``run_backfill`` uncaught.
    """

    async def _raise_cancelled(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", _raise_cancelled)

    code = await run_backfill(phase="all", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 130
    assert "Interrupted" in out
    assert "partial progress was written" in out
    assert "Resume by running" in out
    assert "Backfill summary" in out
    assert "Exit: INTERRUPTED  (130)" in out


def test_backfill_cli_synchronous_keyboard_interrupt_exits_130(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI-level counterpart of the synchronous-raise regression guard.

    Same caveat as its ``run_backfill``-level twin above: this does not
    exercise ``asyncio.Runner``'s SIGINT translation. See
    ``test_real_sigint_during_phase_await_returns_130_with_resume_hint``
    for the real-signal end-to-end path.
    """

    async def _raise_interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", _raise_interrupt)

    result = CliRunner().invoke(
        cascade_mod.app, ["backfill", "--phase", "vectors", "--yes"]
    )

    assert result.exit_code == 130


@pytest.mark.slow
def test_real_sigint_during_phase_await_returns_130_with_resume_hint(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a real terminal Ctrl-C, delivered as an actual SIGINT.

    Exercises the path where on Python 3.11+, ``asyncio.Runner`` (which
    backs ``asyncio.run``, used by the ``backfill`` CLI command)
    converts a real SIGINT into ``main_task.cancel()``, raising
    ``asyncio.CancelledError`` — not ``KeyboardInterrupt`` — at the
    suspended ``await``. A background thread sends a real ``SIGINT`` to
    this test process while a mocked phase is parked in
    ``asyncio.sleep``, reproducing that exact mechanism rather than a
    synchronous stand-in.

    Runs synchronously (not ``async def``): ``pytest-asyncio`` drives
    its tests off a bare event loop via ``run_until_complete``, not
    ``asyncio.Runner``, so it would never install the SIGINT handler
    under test. The CLI command's own ``asyncio.run()`` call is what
    must install it, so this test must invoke the CLI command directly.

    Marked ``slow`` — and therefore excluded from the default
    ``-m "not slow"`` run — because the ``0.3s`` timer that sends the
    signal races ``asyncio.Runner``'s SIGINT-handler installation. On
    a fast host the handler is in place well before ``0.3s`` elapses;
    on a heavily loaded CI runner the signal can land on pytest's
    default handler and kill the test process. Opt-in execution keeps
    the guard available for local repro without destabilising CI.
    """

    async def _slow_phase(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", _slow_phase)

    def _send_sigint_soon() -> None:
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_send_sigint_soon, daemon=True).start()

    result = CliRunner().invoke(
        cascade_mod.app, ["backfill", "--phase", "vectors", "--yes"]
    )

    assert result.exit_code == 130
    assert "Interrupted" in result.stdout
    assert "Resume by running" in result.stdout


# ── --help polish ────────────────────────────────────────────────────────────


def test_backfill_help_documents_every_phase_name() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--help"])
    assert result.exit_code == 0
    for phase_name in ("vectors", "clusters", "skills", "all"):
        assert phase_name in result.stdout


def test_backfill_help_documents_yes_flag_behavior() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--help"])
    assert result.exit_code == 0
    assert "confirm" in result.stdout.lower()


def test_backfill_help_documents_resumption() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--help"])
    assert result.exit_code == 0
    assert "resum" in result.stdout.lower()


def test_backfill_help_documents_cost_estimate_policy() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--help"])
    assert result.exit_code == 0
    assert "token" in result.stdout.lower()
    assert "price" in result.stdout.lower() or "pricing" in result.stdout.lower()
