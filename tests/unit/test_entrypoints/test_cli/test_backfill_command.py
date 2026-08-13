"""``everos cascade backfill`` — scaffold: CLI surface + orchestration skeleton.

Task 20 covers only the phase-iteration / confirmation framework; Task 21
replaced the ``"vectors"`` phase body with the real re-embed
implementation (covered end-to-end in
``tests/integration/test_cli/test_backfill_phase1.py``). This file
stays scoped to the scaffold contract:

- the ``backfill`` subcommand is registered alongside sync/status/fix
- ``--help`` output and ``--phase`` choice validation
- ``run_backfill`` phase filtering (``all`` vs a single slug)
- ``--yes`` / interactive confirmation behaviour, including abort (exit 1)
  and an unexpected-error path (exit 2)

Task 22-23 replace the remaining stub phase bodies (``clusters`` /
``skills``); this file must not assert anything about real
cluster/skill behaviour.

Now that ``"vectors"`` opens real LanceDB tables, every test is
isolated under a tmp memory root (``_isolated_lancedb_root`` below) so
none of them ever touch a developer's real ``~/.everos``.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands import _backfill_cmd as backfill_cmd
from everos.entrypoints.cli.commands import cascade as cascade_mod
from everos.entrypoints.cli.commands._backfill_cmd import PHASES, run_backfill
from everos.infra.persistence.lancedb import Episode, dispose_connection, episode_repo
from everos.memory.cascade import BackfillPhase
from everos.memory.cascade import _backfill as backfill_mod

# rich (via typer) injects ANSI escapes into help output when it thinks
# the terminal supports colour; on CI those escapes chop `--phase` into
# `-\x1b[…m-phase\x1b[…m`, defeating a plain substring match. Strip them
# before asserting on flag names.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def _isolated_lancedb_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """The ``"vectors"`` phase now opens real LanceDB tables to scan for
    ``vector IS NULL`` rows — isolate every test under a tmp memory root
    so none of them touch a developer's real ``~/.everos``, and dispose
    the cached connection around each test so a stale handle from a
    previous tmp_path never leaks into the next one (mirrors
    ``cli_runtime`` in ``test_cascade_cli_integration.py``).
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    asyncio.run(dispose_connection())
    yield
    asyncio.run(dispose_connection())


class _StubAvailableCapability:
    """Round-2 finding #5 moved capability preflight to BEFORE every
    phase's collection scan (including the ``total == 0`` early
    return). Tests that used to reach the "nothing to backfill" green
    path with the default (unavailable) capability would now
    short-circuit to exit 2. Presenting an available capability
    everywhere in this file preserves the scaffold-contract intent —
    these tests never care about the embedding branch.
    """

    available = True

    def require(self):
        # None of these tests actually run a real embed batch — the
        # phase bodies scan an empty backlog and return before touching
        # the provider. Returning ``None`` keeps the mock cheap.
        return None


@pytest.fixture(autouse=True)
def _preflight_capability_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the embedding capability to available so every phase's
    round-2 preflight passes and control reaches the scaffold code
    these tests actually exercise."""
    monkeypatch.setattr(
        backfill_mod, "get_embedding_capability", lambda: _StubAvailableCapability()
    )


async def _seed_one_unbackfilled_episode() -> None:
    """One ``vector IS NULL`` row so a declined-confirmation test actually
    reaches the confirm prompt instead of short-circuiting on "nothing to
    backfill"."""
    await episode_repo.add(
        [
            Episode(
                id="u1_ep1",
                entry_id="ep1",
                owner_id="u1",
                owner_type="user",
                timestamp=get_utc_now(),
                parent_id="mc1",
                sender_ids=["u1"],
                episode="episode body",
                episode_tokens="episode body",
                md_path="users/u1/episodes/episode-2026-01-01.md",
                content_sha256=hashlib.sha256(b"ep1").hexdigest(),
                vector=None,
            )
        ]
    )


def test_backfill_help_exits_zero() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--help"])
    assert result.exit_code == 0
    stdout = _strip_ansi(result.stdout)
    assert "--phase" in stdout
    assert "--yes" in stdout


def test_backfill_invalid_phase_rejected() -> None:
    result = CliRunner().invoke(cascade_mod.app, ["backfill", "--phase", "bogus"])
    assert result.exit_code != 0


def test_phases_slugs_match_cli_choices() -> None:
    assert [p.slug for p in PHASES] == ["vectors", "clusters", "skills"]


async def test_run_backfill_single_phase_runs_only_that_phase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await run_backfill(phase="clusters", auto_yes=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "Phase 2" in out
    assert "Phase 1" not in out
    assert "Phase 3" not in out


async def test_run_backfill_all_runs_every_phase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await run_backfill(phase="all", auto_yes=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "Phase 1" in out
    assert "Phase 2" in out
    assert "Phase 3" in out


async def test_run_backfill_yes_skips_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("typer.confirm must not be called when auto_yes=True")

    monkeypatch.setattr(typer, "confirm", _boom)
    code = await run_backfill(phase="vectors", auto_yes=True)
    assert code == 0


async def test_run_backfill_abort_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User declines the y/N prompt → exit 1 (aborted).

    Preflight now checks the embedding capability BEFORE the confirm
    prompt (Group A / finding #1), so this test has to make capability
    available or the phase short-circuits with exit 2
    (``blocked_by_capability``) before ever reaching ``typer.confirm``.
    Provided in-file rather than via the shared fixture in
    ``test_backfill_preflight.py`` because these tests live under
    ``test_entrypoints/`` and don't pull that fixture in.
    """
    await _seed_one_unbackfilled_episode()

    class _AvailableCapability:
        available = True

        def require(self):
            raise AssertionError(
                "require() should not run: the test declines the confirm "
                "prompt before the phase body ever asks for the provider."
            )

    monkeypatch.setattr(
        backfill_mod, "get_embedding_capability", lambda: _AvailableCapability()
    )
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    code = await run_backfill(phase="vectors", auto_yes=False)
    assert code == 1


async def test_run_backfill_unexpected_error_returns_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(backfill_cmd, "_print_phase_header", _raise)
    code = await run_backfill(phase="vectors", auto_yes=True)
    assert code == 2


def test_confirm_auto_yes_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("must not prompt when auto_yes=True")

    monkeypatch.setattr(typer, "confirm", _boom)
    assert backfill_cmd._confirm("anything", auto_yes=True) is True


def test_confirm_delegates_to_typer_when_not_auto_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    assert backfill_cmd._confirm("anything", auto_yes=False) is True


def test_backfill_cli_abort_via_n_returns_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-driven abort mirror of :func:`test_run_backfill_abort_returns_one`.

    Same reasoning: preflight now runs before the confirm prompt, so
    the test must present an available capability or the "n" input
    never reaches ``_confirm``.
    """
    asyncio.run(_seed_one_unbackfilled_episode())

    class _AvailableCapability:
        available = True

        def require(self):
            raise AssertionError("require() should not run in an abort test")

    monkeypatch.setattr(
        backfill_mod, "get_embedding_capability", lambda: _AvailableCapability()
    )
    result = CliRunner().invoke(
        cascade_mod.app, ["backfill", "--phase", "vectors"], input="n\n"
    )
    assert result.exit_code == 1


def test_backfill_cli_yes_flag_auto_confirms() -> None:
    result = CliRunner().invoke(
        cascade_mod.app, ["backfill", "--phase", "clusters", "--yes"]
    )
    assert result.exit_code == 0
    assert "Phase 2" in result.stdout


def test_backfill_phase_dataclass_is_frozen() -> None:
    phase = PHASES[0]
    assert isinstance(phase, BackfillPhase)
    with pytest.raises(Exception):  # noqa: B017 — dataclasses.FrozenInstanceError
        phase.number = 99  # type: ignore[misc]
