"""Tests for :func:`reconcile` — pure scan vs state diff.

The reconciler is pure (no IO), so each scenario is just a few
dataclass instances in / decisions out. Covers the 4 cases:
``added`` / ``modified`` / ``deleted`` / ``no-op``.
"""

from __future__ import annotations

from everos.memory.cascade.reconciler import PriorState, reconcile
from everos.memory.cascade.types import ScanInput


def _scan(path: str, mtime: float = 1.0, kind: str = "episode") -> ScanInput:
    return ScanInput(md_path=path, mtime=mtime, kind=kind)


def _state(
    path: str,
    *,
    mtime: float = 1.0,
    kind: str = "episode",
    status: str = "done",
    change_type: str = "modified",
) -> PriorState:
    return PriorState(
        md_path=path,
        kind=kind,
        mtime=mtime,
        status=status,
        change_type=change_type,
    )


def test_added_path_emits_added_decision() -> None:
    decisions = reconcile([_scan("a.md")], state={})
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "added")]


def test_modified_mtime_emits_modified_decision() -> None:
    decisions = reconcile(
        [_scan("a.md", mtime=2.0)],
        state={"a.md": _state("a.md", mtime=1.0)},
    )
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "modified")]


def test_done_state_with_matching_mtime_is_skipped() -> None:
    """Quiet sweeps must stay quiet — no upsert churn."""
    decisions = reconcile(
        [_scan("a.md", mtime=1.0)],
        state={"a.md": _state("a.md", mtime=1.0, status="done")},
    )
    assert decisions == []


def test_pending_state_with_matching_mtime_still_emits_modified() -> None:
    """Pending / failed states are NOT terminal — re-emit so worker re-runs."""
    decisions = reconcile(
        [_scan("a.md", mtime=1.0)],
        state={"a.md": _state("a.md", mtime=1.0, status="pending")},
    )
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "modified")]


def test_deleted_path_emits_deleted_decision() -> None:
    decisions = reconcile(
        [],
        state={"a.md": _state("a.md", status="pending")},
    )
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "deleted")]


def test_deleted_path_already_done_as_delete_is_skipped() -> None:
    """A done row that is itself a successful delete cycle — don't re-emit."""
    decisions = reconcile(
        [],
        state={
            "a.md": _state("a.md", status="done", change_type="deleted"),
        },
    )
    assert decisions == []


def test_done_added_row_with_missing_path_is_recovered_as_deleted() -> None:
    """Watcher missed an unlink (e.g. fseventsd drop / daemon restart).

    The state row is ``status='done'`` from the previous add cycle, but
    the file is gone from disk. The scanner MUST re-emit a 'deleted'
    decision — otherwise LanceDB keeps stale rows for the orphan path
    until something else triggers an enqueue.
    """
    decisions = reconcile(
        [],
        state={
            "a.md": _state("a.md", status="done", change_type="added"),
        },
    )
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "deleted")]


def test_done_modified_row_with_missing_path_is_recovered_as_deleted() -> None:
    """Same as the added variant, but the prior cycle was a modification."""
    decisions = reconcile(
        [],
        state={
            "a.md": _state("a.md", status="done", change_type="modified"),
        },
    )
    assert [(d.md_path, d.change_type) for d in decisions] == [("a.md", "deleted")]


def test_mixed_scenario_preserves_order() -> None:
    decisions = reconcile(
        [
            _scan("new.md"),
            _scan("changed.md", mtime=2.0),
            _scan("unchanged.md", mtime=1.0),
        ],
        state={
            "changed.md": _state(
                "changed.md", mtime=1.0, status="done", change_type="modified"
            ),
            "unchanged.md": _state(
                "unchanged.md", mtime=1.0, status="done", change_type="modified"
            ),
            "gone.md": _state("gone.md", status="pending", change_type="modified"),
        },
    )
    by_path = {d.md_path: d.change_type for d in decisions}
    assert by_path == {
        "new.md": "added",
        "changed.md": "modified",
        "gone.md": "deleted",
    }
    # Order: added/modified in scan order, deleted at the tail.
    assert decisions[-1].md_path == "gone.md"
