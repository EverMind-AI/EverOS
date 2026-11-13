"""The delete fence: index rows a deletion removed, kept from coming back.

A cascade run reads an md file, embeds the entries that changed and then
upserts them, with no lock between the read and the write. A deletion that
rewrites the file and deletes the index rows in that gap would see its rows
written back by a run that read the file before the deletion. So the
deletion first adds the rows' keys to this fence, then deletes them; a run
drops fenced rows before its upsert and checks the fence again after it,
deleting any row a deletion fenced meanwhile.

Both steps are synchronous, so no lock is needed: if the deletion fenced a
row before the run's check after the upsert, the run deletes it; if after,
the deletion's own delete was issued after the upsert returned.

An entry is kept only while a run that started before it is still going:
every run registers when it starts, before it reads the file, and
unregisters when it ends (success, failure or timeout alike). A run that
starts after the deletion reads the rewritten file, which no longer has the
entry. With no run in flight a deletion keeps nothing. A restart empties the
fence and loses nothing: the runs in flight ended with the process.
"""

from __future__ import annotations

__all__ = ["DeleteFence"]

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Gauge
from everos.memory.ports import Filter, all_of, any_of, eq, one_of

logger = get_logger(__name__)

type RowKey = tuple[str, str, str]
"""``(app_id, project_id, id)``: the primary key of every daily-log table."""

_entries_gauge = Gauge(
    name="memory_delete_fence_entries",
    description="Index rows a deletion fenced off from in-flight cascade runs.",
    namespace="everos",
)
_runs_gauge = Gauge(
    name="memory_delete_fence_runs",
    description="Cascade runs in flight, as registered with the delete fence.",
    namespace="everos",
)


class DeleteFence:
    """Row keys removed by deletions, while earlier cascade runs are in flight.

    Args:
        warn_entries: Log ``memory_delete_fence_large`` once the fence holds
            more entries than this (``[cascade] delete_fence_warn_entries``).
    """

    def __init__(self, *, warn_entries: int = 100_000) -> None:
        self._warn_entries = warn_entries
        self._seq = 0
        self._runs: set[int] = set()
        self._entries: dict[RowKey, int] = {}
        self._publish()

    def register_run(self) -> int:
        """A cascade run starts (before it reads its file); returns its token."""
        self._seq += 1
        self._runs.add(self._seq)
        self._publish()
        return self._seq

    def end_run(self, token: int) -> None:
        """A cascade run ended, however it ended."""
        self._runs.discard(token)
        self._prune()

    def add(self, keys: Iterable[RowKey]) -> None:
        """Fence the rows a deletion is about to delete from the index."""
        if not self._runs:
            return  # every later run reads the rewritten file
        self._seq += 1
        before = len(self._entries)
        for key in keys:
            self._entries[key] = self._seq
        self._prune()
        if before <= self._warn_entries < len(self._entries):
            logger.warning(
                "memory_delete_fence_large",
                entries=len(self._entries),
                runs=len(self._runs),
                threshold=self._warn_entries,
            )

    def in_flight(self) -> frozenset[int]:
        """Tokens of the runs registered right now."""
        return frozenset(self._runs)

    def blocks(self, row: Any) -> bool:
        """Whether ``row`` (an index record) was fenced by a deletion."""
        return (row.app_id, row.project_id, row.id) in self._entries

    def filter_for(self, rows: Sequence[Any]) -> Filter:
        """The index predicate that deletes exactly ``rows``."""
        ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in rows:
            ids[(row.app_id, row.project_id)].append(row.id)
        return any_of(
            *(
                all_of(eq("app_id", app), eq("project_id", project), one_of("id", rid))
                for (app, project), rid in ids.items()
            )
        )

    def _prune(self) -> None:
        """Drop the entries no run in flight could have read past."""
        oldest = min(self._runs, default=None)
        if oldest is None:
            self._entries.clear()
        else:
            self._entries = {k: s for k, s in self._entries.items() if s > oldest}
        self._publish()

    def _publish(self) -> None:
        _entries_gauge.set(len(self._entries))
        _runs_gauge.set(len(self._runs))
