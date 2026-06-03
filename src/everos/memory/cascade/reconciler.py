"""Reconcile scanner observations against the current ``md_change_state``.

Pure function — given a list of :class:`ScanInput` (what's on disk
right now, matched against the kind registry) and the prior state
keyed by ``md_path`` (what's in sqlite), emit the
:class:`ReconcileDecision` set the scanner should UPSERT.

Three categories per 12 doc §5.3:

- **added** — path on disk, no prior state row.
- **modified** — path on disk, prior mtime differs (newer or older;
  cascade does not assume monotonic file timestamps).
- **deleted** — path missing from disk. Emitted unless the prior
  row is *already* the result of a successful delete cycle
  (``status='done' AND change_type='deleted'`` — handler has wiped
  the orphan LanceDB rows on a previous sweep). A ``status='done'``
  row whose ``change_type`` is ``'added'`` / ``'modified'`` is still
  emitted: the watcher must have missed the unlink (e.g. fseventsd
  drop, daemon restart window), and we need the scanner to recover
  the deletion or LanceDB stays stale.

Paths whose prior state row is ``done`` AND the mtime matches are
skipped on the add/modify side — the reconcile output stays tight on
quiet sweeps.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

from .types import ReconcileDecision, ScanInput


@dataclasses.dataclass(frozen=True)
class PriorState:
    """Snapshot of one ``md_change_state`` row, as far as reconcile cares.

    Only the fields needed to decide change_type appear here; the
    repo's full :class:`MdChangeState` row is reduced to this shape
    at the reconciler entry boundary.
    """

    md_path: str
    kind: str
    mtime: float
    status: str  # "pending" | "processing" | "done" | "failed"
    change_type: str  # "added" | "modified" | "deleted"


def reconcile(
    scan: Iterable[ScanInput],
    state: Mapping[str, PriorState],
) -> list[ReconcileDecision]:
    """Compute the UPSERT plan for one scanner sweep.

    Args:
        scan: Every md path currently on disk that matches a registered
            kind (scanner output).
        state: ``{md_path: PriorState}`` — the current
            ``md_change_state`` snapshot keyed by path.

    Returns:
        Ordered :class:`ReconcileDecision` list — ``added`` /
        ``modified`` first (in scan order), then ``deleted`` for paths
        present in state but missing from disk.
    """
    decisions: list[ReconcileDecision] = []
    seen: set[str] = set()

    for item in scan:
        seen.add(item.md_path)
        prior = state.get(item.md_path)
        if prior is None:
            decisions.append(
                ReconcileDecision(
                    md_path=item.md_path,
                    kind=item.kind,
                    change_type="added",
                    mtime=item.mtime,
                )
            )
            continue
        # Skip when the row is already done and mtime hasn't moved.
        if prior.status == "done" and prior.mtime == item.mtime:
            continue
        decisions.append(
            ReconcileDecision(
                md_path=item.md_path,
                kind=item.kind,
                change_type="modified",
                mtime=item.mtime,
            )
        )

    for path, prior in state.items():
        if path in seen:
            continue
        # File missing from disk. Only skip when the prior cycle was
        # *itself* a successful delete (the handler already wiped the
        # orphan LanceDB rows). A done row with change_type='added' /
        # 'modified' means the watcher missed the subsequent unlink —
        # without re-emitting 'deleted' here the scanner would never
        # recover the stale LanceDB rows.
        if prior.status == "done" and prior.change_type == "deleted":
            continue
        decisions.append(
            ReconcileDecision(
                md_path=path,
                kind=prior.kind,
                change_type="deleted",
                mtime=prior.mtime,
            )
        )

    return decisions
