"""Preflight tests for backfill Phase 1/2/3 short-circuits.

Pins two friendly-error contracts added for the embed-soft-dependency
e2e polish (spec §10 revision):

1. Missing capability — when the required provider (currently only
   ``embedding``) is not configured, every phase must short-circuit
   BEFORE the y/N prompt with a toml-hint message, and ``run_backfill``
   must map that to exit code ``2`` (``FAILED``) rather than exit ``1``
   ("aborted by user") or the generic "see logs for details" bucket.
2. Server-running detection — when another process holds
   ``ome.db.lock``, Phase 2 (clusters) and Phase 3 (skills) must
   short-circuit before ``engine.start()`` fires and before the user is
   asked to confirm, returning a result flagged ``blocked_by_server``.
   ``run_backfill`` then maps that to exit code ``3``
   (``SERVER_RUNNING``) — never surfacing a raw ``EngineLockHeldError``
   traceback.

Preflight order inside each phase is capability → OME lock: a permanent
config gap wins over a transient lock condition.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from everos.entrypoints.cli.commands import _backfill_cmd
from everos.memory.cascade import _backfill
from everos.memory.cascade._backfill import NullBackfillPresenter


class _FailingConfirmPresenter(NullBackfillPresenter):
    """Presenter whose :meth:`confirm` raises — used by preflight tests
    to assert control never reaches the y/N prompt."""

    async def confirm(self, prompt: str, *, auto_yes: bool) -> bool:
        raise AssertionError(
            "presenter.confirm called after a preflight blocker — capability "
            "check must short-circuit before the y/N prompt"
        )


class _FakeCapabilityAvailable:
    """Stand-in for :class:`EmbeddingCapability` with ``available = True``.

    Preflight now checks capability BEFORE the OME lock, so any test
    that exercises the lock branch must first pass the capability
    branch. Providing a truthy capability keeps the tests focused on
    the branch under test.
    """

    available = True

    def require(self):
        return None


@pytest.fixture
def _isolated_root(tmp_path: Path, monkeypatch) -> Path:
    """Point ``MemoryRoot.resolve()`` at a per-test temp directory.

    Preflight calls ``MemoryRoot.resolve()`` — even when we mock the probe
    result — because helper docstrings promise ``lock_path.parent.mkdir``
    doesn't blow up on a real tree, and future assertions may inspect
    the resolved root. Isolating it keeps the test off ``~/.everos``.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def _capability_available(monkeypatch) -> None:
    """Mock ``get_embedding_capability`` to available inside ``_backfill``.

    The default test settings leave ``[embedding].api_key`` blank, so
    the real accessor returns an unavailable capability — which would
    now trip the capability preflight before any other branch. Every
    non-capability-branch test opts into this fixture to bypass that.
    """
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityAvailable()
    )


async def _one_episode_row() -> list[dict[str, object]]:
    """Return a single synthetic episode row shaped for
    :func:`_scan_all_rows`. Shared by the blocked_by_server tests so
    the phase runs past the "nothing to backfill" early return and
    actually reaches the preflight branch."""
    return [
        {
            "parent_type": "memcell",
            "parent_id": "mc_1",
            "entry_id": "ep_1",
            "episode": "hello",
            "timestamp": 1_700_000_000_000,
            "owner_id": "alice",
            "app_id": "default",
            "project_id": "default",
        }
    ]


def _patch_phase2_scans_have_data(monkeypatch) -> None:
    """Wire ``_scan_all_rows`` to a stub returning one Episode row.

    Preflight now runs AFTER the initial scan (per the phase runner's
    new order), so a test that isolates the OME-lock branch must ensure
    the scan does not short-circuit with "nothing to backfill" — the
    fixture guarantees at least one row is visible.
    """
    from everos.infra.persistence.lancedb import Episode

    async def _fake_scan_all_rows(schema):  # type: ignore[no-untyped-def]
        if schema is Episode:
            return await _one_episode_row()
        return []

    monkeypatch.setattr(_backfill, "_scan_all_rows", _fake_scan_all_rows)


def _patch_phase3_scans_have_data(monkeypatch) -> None:
    """Same idea for Phase 3: seed one ``_SkillSourceRow`` so the phase
    reaches the preflight instead of hitting "nothing to backfill"."""

    async def _fake_ensure_schema() -> None:
        return None

    async def _fake_scan_skill_source():
        return [
            _backfill._SkillSourceRow(
                case_entry_id="case_1",
                cluster_id="c_1",
                agent_id="agent_a",
                app_id="default",
                project_id="default",
            )
        ]

    monkeypatch.setattr(_backfill, "_ensure_cluster_schema", _fake_ensure_schema)
    monkeypatch.setattr(_backfill, "_scan_skill_source", _fake_scan_skill_source)


async def test_phase_vectors_preflight_returns_blocked_by_server(
    _isolated_root: Path, _capability_available: None, monkeypatch
) -> None:
    """Phase 1 preflight parity with Phase 2/3.

    Regression: prior to PR #361 review J10, ``_run_phase_vectors``
    only preflighted capability, not the OME lock. Running
    ``--phase all`` against a live server would burn Phase 1's full
    embed API cost before Phase 2 finally halted with exit 3. Phase 1
    must fail fast BEFORE any provider call.
    """

    async def _fake_scan():
        # Scan must never run — preflight should short-circuit first.
        raise AssertionError("Phase 1 scan ran despite blocked server")

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    result = await _backfill._run_phase_vectors(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.aborted is True
    assert result.blocked_by_server is True
    assert result.rows_processed == 0


async def test_run_backfill_phase_vectors_exits_3_when_blocked(
    _isolated_root: Path, _capability_available: None, monkeypatch, capsys
) -> None:
    """``run_backfill --phase vectors`` maps blocked_by_server → exit 3."""

    async def _fake_scan():
        raise AssertionError("Phase 1 scan ran despite blocked server")

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    exit_code = await _backfill_cmd.run_backfill(phase="vectors", auto_yes=True)
    assert exit_code == 3


async def test_phase_clusters_preflight_returns_blocked_by_server(
    _isolated_root: Path, _capability_available: None, monkeypatch
) -> None:
    """Phase 2 preflight: probe False → aborted + blocked_by_server."""
    _patch_phase2_scans_have_data(monkeypatch)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    result = await _backfill._run_phase_clusters(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.aborted is True
    assert result.blocked_by_server is True
    assert result.events_emitted == 0


async def test_phase_skills_preflight_returns_blocked_by_server(
    _isolated_root: Path, _capability_available: None, monkeypatch
) -> None:
    """Phase 3 preflight mirrors Phase 2."""
    _patch_phase3_scans_have_data(monkeypatch)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    result = await _backfill._run_phase_skills(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.aborted is True
    assert result.blocked_by_server is True
    assert result.events_emitted == 0


async def test_run_backfill_exits_3_when_clusters_blocked(
    _isolated_root: Path, _capability_available: None, monkeypatch, capsys
) -> None:
    """``run_backfill`` maps blocked_by_server to exit code 3 and prints
    the friendly hint without a python traceback."""
    _patch_phase2_scans_have_data(monkeypatch)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    exit_code = await _backfill_cmd.run_backfill(phase="clusters", auto_yes=True)
    assert exit_code == 3

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Stop your everos server first" in combined
    assert "SERVER_RUNNING" in combined
    # Guard against regressing to the raw-traceback path. We name the
    # exception class explicitly rather than the generic "Traceback"
    # marker — other, unrelated tests in the suite occasionally emit a
    # stdlib "Logging error" traceback when a captured stream closes,
    # and that flakes any generic-string check without indicating a
    # real regression here.
    assert "EngineLockHeldError" not in combined


async def test_run_backfill_exits_3_when_skills_blocked(
    _isolated_root: Path, _capability_available: None, monkeypatch, capsys
) -> None:
    """Same exit-code mapping for Phase 3."""
    _patch_phase3_scans_have_data(monkeypatch)
    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: False)
    exit_code = await _backfill_cmd.run_backfill(phase="skills", auto_yes=True)
    assert exit_code == 3

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Stop your everos server first" in combined
    assert "EngineLockHeldError" not in combined


async def test_engine_start_race_still_reports_blocked(
    _isolated_root: Path, monkeypatch
) -> None:
    """Probe succeeded, but a server started between probe and
    ``engine.start()``. The phase must still return blocked_by_server
    rather than leaking :class:`EngineLockHeldError`.
    """
    from everos.infra.ome.exceptions import EngineLockHeldError

    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", lambda: True)

    class _FakeEngine:
        async def start(self) -> None:
            raise EngineLockHeldError("simulated race")

        async def stop(self) -> None:
            return None

    # Force the phase past the "nothing to backfill" early return by
    # pretending one episode row exists so we exercise the engine.start()
    # code path.
    async def _fake_scan_all_rows(schema):  # type: ignore[no-untyped-def]
        from everos.infra.persistence.lancedb import Episode

        if schema is Episode:
            return [
                {
                    "parent_type": "memcell",
                    "parent_id": "mc_1",
                    "entry_id": "ep_1",
                    "episode": "hello",
                    "timestamp": 1_700_000_000_000,
                    "owner_id": "alice",
                    "app_id": "default",
                    "project_id": "default",
                }
            ]
        return []

    monkeypatch.setattr(_backfill, "_scan_all_rows", _fake_scan_all_rows)

    class _FakeCapability:
        available = True

        def require(self):
            return None

    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapability()
    )

    async def _fake_ensure_schema() -> None:
        return None

    monkeypatch.setattr(_backfill, "_ensure_cluster_schema", _fake_ensure_schema)

    class _FakeClusterRepo:
        async def count(self) -> int:
            return 0

    monkeypatch.setattr(_backfill, "cluster_repo", _FakeClusterRepo())
    monkeypatch.setattr(_backfill, "_build_cluster_engine", lambda: _FakeEngine())

    # This test drives past the confirm prompt (auto_yes=True is
    # honoured by ``NullBackfillPresenter``) to exercise the
    # engine.start() race branch — swapping in the failing presenter
    # would misrepresent the branch under test.
    result = await _backfill._run_phase_clusters(
        auto_yes=True, presenter=NullBackfillPresenter()
    )
    assert result.aborted is True
    assert result.blocked_by_server is True


def test_exit_labels_include_server_running() -> None:
    """``_EXIT_LABELS`` gains the 3 = SERVER_RUNNING mapping without
    perturbing existing codes."""
    assert _backfill_cmd._EXIT_LABELS[3] == "SERVER_RUNNING"
    assert _backfill_cmd._EXIT_LABELS[0] == "SUCCESS"
    assert _backfill_cmd._EXIT_LABELS[1] == "ABORTED"
    assert _backfill_cmd._EXIT_LABELS[2] == "FAILED"
    assert _backfill_cmd._EXIT_LABELS[130] == "INTERRUPTED"


# ── capability preflight (finding #1) ─────────────────────────────────────────


class _FakeCapabilityMissing:
    """Stand-in for :class:`EmbeddingCapability` with ``available = False``."""

    available = False

    def require(self):
        from everos.core.errors import ProviderNotConfiguredError

        raise ProviderNotConfiguredError(provider="embedding")


async def test_phase_vectors_preflight_capability_missing(
    _isolated_root: Path, monkeypatch
) -> None:
    """Phase 1: capability missing → ``blocked_by_capability='embedding'``
    without calling ``_confirm``."""
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    # Force the scan to look like there's work to do so we don't take the
    # "nothing to backfill" early-return path (which short-circuits BEFORE
    # preflight — that path is unaffected by this fix).
    async def _fake_scan():
        from everos.memory.cascade._backfill import (
            _TABLE_SPECS,
            _NullVectorRow,
            _TableBacklog,
        )

        return [
            _TableBacklog(
                spec=_TABLE_SPECS[0],
                rows=[
                    _NullVectorRow(
                        id="row_1",
                        text="hello",
                        subject_text=None,
                        tokens=5,
                    )
                ],
            )
        ], 0

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)

    result = await _backfill._run_phase_vectors(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.aborted is False
    assert result.rows_processed == 0


async def test_phase_clusters_preflight_capability_missing(
    _isolated_root: Path, monkeypatch
) -> None:
    """Phase 2: capability missing → ``blocked_by_capability='embedding'``
    without calling ``_confirm`` and without probing the OME lock."""
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    def _probe_should_not_run() -> bool:
        raise AssertionError(
            "OME lock probe ran after capability preflight failed — capability "
            "is a permanent gap; probing the transient lock adds nothing."
        )

    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", _probe_should_not_run)

    async def _fake_scan_all_rows(schema):  # type: ignore[no-untyped-def]
        from everos.infra.persistence.lancedb import Episode

        if schema is Episode:
            return [
                {
                    "parent_type": "memcell",
                    "parent_id": "mc_1",
                    "entry_id": "ep_1",
                    "episode": "hello",
                    "timestamp": 1_700_000_000_000,
                    "owner_id": "alice",
                    "app_id": "default",
                    "project_id": "default",
                }
            ]
        return []

    monkeypatch.setattr(_backfill, "_scan_all_rows", _fake_scan_all_rows)

    result = await _backfill._run_phase_clusters(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.blocked_by_server is False
    assert result.aborted is False


async def test_phase_skills_preflight_capability_missing(
    _isolated_root: Path, monkeypatch
) -> None:
    """Phase 3: capability missing → ``blocked_by_capability='embedding'``."""
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    def _probe_should_not_run() -> bool:
        raise AssertionError("OME lock probe ran after capability preflight failed")

    monkeypatch.setattr(_backfill, "_probe_ome_lock_available", _probe_should_not_run)

    async def _fake_ensure_schema() -> None:
        return None

    monkeypatch.setattr(_backfill, "_ensure_cluster_schema", _fake_ensure_schema)

    async def _fake_scan_skill_source():
        from everos.memory.cascade._backfill import _SkillSourceRow

        return [
            _SkillSourceRow(
                case_entry_id="case_1",
                cluster_id="c_1",
                agent_id="agent_a",
                app_id="default",
                project_id="default",
            )
        ]

    monkeypatch.setattr(_backfill, "_scan_skill_source", _fake_scan_skill_source)

    result = await _backfill._run_phase_skills(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.blocked_by_server is False
    assert result.aborted is False


async def test_run_backfill_returns_2_when_blocked_by_capability(
    _isolated_root: Path, monkeypatch, capsys
) -> None:
    """``run_backfill`` maps blocked_by_capability → exit 2 and the summary
    reports ``FAILED``. Prior to the fix, the same condition raised
    ``ProviderNotConfiguredError`` mid-phase, which the broad
    ``except Exception`` swallowed into "Backfill failed — see logs".
    """

    async def _fake_phase_vectors(*, auto_yes: bool, presenter: object):
        return _backfill._PhaseResult(blocked_by_capability="embedding")

    monkeypatch.setattr(_backfill, "_run_phase_vectors", _fake_phase_vectors)

    exit_code = await _backfill_cmd.run_backfill(phase="vectors", auto_yes=True)
    assert exit_code == 2

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _backfill_cmd._EXIT_LABELS[2] == "FAILED"
    assert "FAILED" in combined
    # Regression guard: the phase's own toml-hint output must not be
    # swallowed by the generic-error path.
    assert "Backfill failed — see logs for details." not in combined


async def test_run_backfill_returns_2_and_prints_toml_hint_end_to_end(
    _isolated_root: Path, monkeypatch, capsys
) -> None:
    """End-to-end: real phase runner (capability unavailable via a fake
    accessor) prints the toml-hint copy and ``run_backfill`` returns 2.
    """
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    async def _fake_scan():
        from everos.memory.cascade._backfill import (
            _TABLE_SPECS,
            _NullVectorRow,
            _TableBacklog,
        )

        return [
            _TableBacklog(
                spec=_TABLE_SPECS[0],
                rows=[
                    _NullVectorRow(
                        id="row_1",
                        text="hello",
                        subject_text=None,
                        tokens=5,
                    )
                ],
            )
        ], 0

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)

    exit_code = await _backfill_cmd.run_backfill(phase="vectors", auto_yes=True)
    assert exit_code == 2

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # Copy is stable via missing_config_error(); look for the invariant
    # substring (toml section + everos init hint).
    assert "everos.toml" in combined
    assert "[embedding]" in combined
    assert "FAILED" in combined


# ── Ctrl-C at confirm (finding #2) ───────────────────────────────────────────


# ── round-2 finding #5: preflight BEFORE scan/tokenize ──────────────────────


async def test_phase_vectors_preflight_before_scan(
    _isolated_root: Path, monkeypatch
) -> None:
    """Round-2 finding #5: capability preflight must run BEFORE
    :func:`_scan_null_vector_backlog`. A user with a 500k-row backlog
    and no embedding configured should exit fast with the toml hint,
    not pay the full O(N) scan + tokenize first.
    """
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    async def _scan_should_not_run():  # type: ignore[no-untyped-def]
        raise AssertionError(
            "_scan_null_vector_backlog ran before capability preflight — "
            "large backlogs would pay an O(N) scan the user can't act on"
        )

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _scan_should_not_run)

    result = await _backfill._run_phase_vectors(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.rows_processed == 0
    assert result.rows_failed == 0


async def test_phase_vectors_empty_backlog_still_preflights(
    _isolated_root: Path, monkeypatch
) -> None:
    """Even when the scan would return zero rows, missing capability
    must yield the toml hint rather than a bare "nothing to backfill"
    green message. Preflight fires first, so we never reach the scan.
    """
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    scan_called = {"count": 0}

    async def _empty_scan():  # type: ignore[no-untyped-def]
        scan_called["count"] += 1
        return [], 0

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _empty_scan)

    result = await _backfill._run_phase_vectors(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    # Preflight-first means the scan is never invoked — a fresh install
    # sees the toml hint, not the "nothing to backfill" copy.
    assert scan_called["count"] == 0


async def test_phase_clusters_preflight_before_scan(
    _isolated_root: Path, monkeypatch
) -> None:
    """Phase 2 mirrors Phase 1: capability preflight fires before
    ``_scan_all_rows`` walks Episode + AgentCase."""
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    async def _scan_should_not_run(schema):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "_scan_all_rows ran before capability preflight in Phase 2"
        )

    monkeypatch.setattr(_backfill, "_scan_all_rows", _scan_should_not_run)

    result = await _backfill._run_phase_clusters(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.aborted is False


async def test_phase_skills_preflight_before_scan(
    _isolated_root: Path, monkeypatch
) -> None:
    """Phase 3 mirrors Phase 1/2 — capability check gates the cluster
    scan."""
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityMissing()
    )

    async def _fake_ensure_schema() -> None:
        return None

    monkeypatch.setattr(_backfill, "_ensure_cluster_schema", _fake_ensure_schema)

    async def _scan_should_not_run():  # type: ignore[no-untyped-def]
        raise AssertionError(
            "_scan_skill_source ran before capability preflight in Phase 3"
        )

    monkeypatch.setattr(_backfill, "_scan_skill_source", _scan_should_not_run)

    result = await _backfill._run_phase_skills(
        auto_yes=True, presenter=_FailingConfirmPresenter()
    )
    assert result.blocked_by_capability == "embedding"
    assert result.aborted is False


async def test_run_backfill_ctrl_c_at_confirm_returns_130(
    _isolated_root: Path, monkeypatch, capsys
) -> None:
    """Ctrl-C at the ``[y/N]`` prompt → typer.confirm re-raises
    ``click.exceptions.Abort`` (a RuntimeError subclass, NOT
    KeyboardInterrupt). ``run_backfill`` must treat it as an interrupt
    (exit 130) and print the resume hint, not fall through to the
    generic ``except Exception`` (exit 2).
    """
    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityAvailable()
    )

    async def _fake_scan():
        from everos.memory.cascade._backfill import (
            _TABLE_SPECS,
            _NullVectorRow,
            _TableBacklog,
        )

        return [
            _TableBacklog(
                spec=_TABLE_SPECS[0],
                rows=[
                    _NullVectorRow(
                        id="row_1",
                        text="hello",
                        subject_text=None,
                        tokens=5,
                    )
                ],
            )
        ], 0

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)

    def _raise_abort(*args, **kwargs):
        raise click.exceptions.Abort()

    monkeypatch.setattr(_backfill_cmd, "_confirm", _raise_abort)

    # auto_yes=False so the _confirm path (our stub) actually fires.
    exit_code = await _backfill_cmd.run_backfill(phase="vectors", auto_yes=False)
    assert exit_code == 130

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Interrupted — partial progress was written." in combined
    assert "INTERRUPTED" in combined
    # Regression guard: must not have taken the generic-error path.
    assert "Backfill failed — see logs for details." not in combined


async def test_run_backfill_typer_abort_at_confirm_returns_130(
    _isolated_root: Path, monkeypatch, capsys
) -> None:
    """Same interrupt semantics as the click-Abort test above, but with
    the real ``typer.Abort`` class typer 0.15+ actually raises.

    Regression: typer vendored click under ``typer._click`` in 0.15+, so
    ``typer.Abort`` and the standalone ``click.exceptions.Abort`` are
    DISTINCT classes — a catch of only ``click.exceptions.Abort`` would
    silently miss the real typer-raised abort (letting it fall through
    to ``except Exception`` → exit 2 with rich traceback). This test
    proves both classes are covered by the interrupt-branch tuple.
    """
    import typer as _typer

    assert _typer.Abort is not click.exceptions.Abort, (
        "typer.Abort must be distinct from click.exceptions.Abort "
        "for this test to exercise the regression scenario"
    )

    monkeypatch.setattr(
        _backfill, "get_embedding_capability", lambda: _FakeCapabilityAvailable()
    )

    async def _fake_scan():
        from everos.memory.cascade._backfill import (
            _TABLE_SPECS,
            _NullVectorRow,
            _TableBacklog,
        )

        return [
            _TableBacklog(
                spec=_TABLE_SPECS[0],
                rows=[
                    _NullVectorRow(
                        id="row_1",
                        text="hello",
                        subject_text=None,
                        tokens=5,
                    )
                ],
            )
        ], 0

    monkeypatch.setattr(_backfill, "_scan_null_vector_backlog", _fake_scan)

    def _raise_typer_abort(*args, **kwargs):
        raise _typer.Abort()

    monkeypatch.setattr(_backfill_cmd, "_confirm", _raise_typer_abort)

    exit_code = await _backfill_cmd.run_backfill(phase="vectors", auto_yes=False)
    assert exit_code == 130

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Interrupted — partial progress was written." in combined
    assert "INTERRUPTED" in combined
    assert "Backfill failed — see logs for details." not in combined
