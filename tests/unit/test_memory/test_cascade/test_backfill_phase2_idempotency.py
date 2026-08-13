"""Phase 2 idempotency across reruns (round-2 finding M2).

``_emit_synthetic_events`` used to fan every episode / agent-case row
into the ephemeral engine unconditionally, so re-running Phase 2 (or
triggering it through ``--phase all`` after a Ctrl-C) re-fired
``EpisodeExtracted`` / ``AgentCaseExtracted`` for rows that a prior
run had already clustered. The clustering strategies have no
event-id dedup at the strategy level and would grow cluster counts
on every rerun — contradicting the CLI's own "nothing is redone from
scratch" promise.

Fix: filter each row through
:meth:`cluster_repo.find_cluster_id_for_member` before emitting. If
the row already sits under a cluster, skip the emit but still count
it toward the progress readout (matches the pre-scan estimate).

The ``member_type`` values used for the lookup (``"episode"`` /
``"case"``) must match what :func:`trigger_profile_clustering` /
:func:`trigger_skill_clustering` insert on the write path — a
mismatch would silently disable the dedup.
"""

from __future__ import annotations

from typing import Any

from everos.component.utils.datetime import get_utc_now
from everos.memory.cascade import _backfill
from everos.memory.cascade._backfill import NullBackfillPresenter
from everos.memory.events import AgentCaseExtracted, EpisodeExtracted


class _RecordingEngine:
    """Spy engine capturing the events actually dispatched."""

    def __init__(self) -> None:
        self.emitted: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.emitted.append(event)


def _episode_row(entry_id: str) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "parent_id": f"mc_{entry_id}",
        "episode": f"body {entry_id}",
        "timestamp": get_utc_now(),
        "owner_id": "u1",
        "session_id": "s1",
        "app_id": "default",
        "project_id": "default",
    }


def _case_row(entry_id: str) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "parent_id": f"mc_{entry_id}",
        "task_intent": f"intent {entry_id}",
        "quality_score": 0.8,
        "timestamp": get_utc_now(),
        "owner_id": "agent1",
        "app_id": "default",
        "project_id": "default",
    }


async def test_emit_synthetic_events_skips_already_clustered_rows(
    monkeypatch,
) -> None:
    """A row whose ``(member_type, entry_id)`` already resolves to a
    cluster is NOT re-emitted — but still counted toward the progress
    tally so the readout matches the pre-scan estimate."""
    lookups: list[tuple[str, str]] = []
    clustered = {
        ("episode", "ep_dup"): "cluster_abc",
        ("case", "ac_dup"): "cluster_def",
    }

    class _StubClusterRepo:
        async def find_cluster_id_for_member(
            self,
            member_type: str,
            member_id: str,
            *,
            app_id: str,
            project_id: str,
            owner_id: str,
        ) -> str | None:
            lookups.append((member_type, member_id))
            return clustered.get((member_type, member_id))

    monkeypatch.setattr(_backfill, "cluster_repo", _StubClusterRepo())

    engine = _RecordingEngine()
    episodes = [_episode_row("ep_fresh"), _episode_row("ep_dup")]
    cases = [_case_row("ac_fresh"), _case_row("ac_dup")]

    processed = await _backfill._emit_synthetic_events(
        engine, episodes, cases, presenter=NullBackfillPresenter()
    )

    # Every row was looked up — dedup must not depend on ordering.
    assert set(lookups) == {
        ("episode", "ep_fresh"),
        ("episode", "ep_dup"),
        ("case", "ac_fresh"),
        ("case", "ac_dup"),
    }

    # Only the fresh rows produced an engine.emit call — the two
    # already-clustered rows were skipped.
    assert len(engine.emitted) == 2
    episode_ids = {
        e.episode_entry_id for e in engine.emitted if isinstance(e, EpisodeExtracted)
    }
    case_ids = {
        e.case_entry_id for e in engine.emitted if isinstance(e, AgentCaseExtracted)
    }
    assert episode_ids == {"ep_fresh"}
    assert case_ids == {"ac_fresh"}

    # Progress count still equals the full input size so the readout
    # matches the pre-scan estimate the user just confirmed.
    assert processed == 4


async def test_emit_synthetic_events_uses_write_path_member_type_strings(
    monkeypatch,
) -> None:
    """Regression guard: the ``member_type`` strings on the lookup path
    (``"episode"`` / ``"case"``) must match what the clustering
    strategies persist on the write path. A drift here would silently
    disable the dedup and re-open the double-cluster window."""
    seen_member_types: set[str] = set()

    class _StubClusterRepo:
        async def find_cluster_id_for_member(
            self,
            member_type: str,
            member_id: str,
            *,
            app_id: str,
            project_id: str,
            owner_id: str,
        ) -> str | None:
            seen_member_types.add(member_type)
            return None

    monkeypatch.setattr(_backfill, "cluster_repo", _StubClusterRepo())

    engine = _RecordingEngine()
    await _backfill._emit_synthetic_events(
        engine,
        [_episode_row("ep1")],
        [_case_row("ac1")],
        presenter=NullBackfillPresenter(),
    )

    # These string constants are pinned by
    # ``trigger_profile_clustering`` (member_type="episode") and
    # ``trigger_skill_clustering`` (member_type="case").
    assert seen_member_types == {"episode", "case"}


async def test_emit_synthetic_events_no_skip_when_no_prior_clusters(
    monkeypatch,
) -> None:
    """Fresh run baseline: every row emits when the reverse index has
    nothing — the fix must not perturb the first-run happy path."""

    class _StubClusterRepo:
        async def find_cluster_id_for_member(
            self,
            member_type: str,
            member_id: str,
            *,
            app_id: str,
            project_id: str,
            owner_id: str,
        ) -> str | None:
            return None

    monkeypatch.setattr(_backfill, "cluster_repo", _StubClusterRepo())

    engine = _RecordingEngine()
    processed = await _backfill._emit_synthetic_events(
        engine,
        [_episode_row("ep1"), _episode_row("ep2")],
        [_case_row("ac1")],
        presenter=NullBackfillPresenter(),
    )

    assert processed == 3
    assert len(engine.emitted) == 3
