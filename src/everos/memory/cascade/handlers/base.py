"""Handler chassis — abstract base + shared :class:`HandlerDeps`.

Each kind has its own concrete handler responsible for translating a
single md path's content (or absence) into the corresponding LanceDB
row state. The handler is the *only* cascade-side piece that knows
the per-kind row shape; everything around it (watcher / scanner /
worker / orchestrator / CLI) is kind-agnostic.

Per-kind handlers share the same dependencies bundle — embedding,
tokenizer, memory-root path resolver — packaged as
:class:`HandlerDeps`. Construct it once at orchestrator startup, pass
to every handler factory, no per-row resolution churn.
"""

from __future__ import annotations

import abc
import dataclasses

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot

from ..types import HandlerOutcome


@dataclasses.dataclass(frozen=True)
class HandlerDeps:
    """Shared providers handed to every :class:`Handler` on construction.

    Frozen + read-only — handlers never mutate the deps; orchestrator
    is the sole owner. ``memory_root`` is used to resolve the absolute
    file path from the relative ``md_path`` stored in
    ``md_change_state``.
    """

    memory_root: MemoryRoot
    embedder: EmbeddingProvider
    tokenizer: Tokenizer


class Handler(abc.ABC):
    """Per-kind cascade handler contract.

    ``handle_added_or_modified`` and ``handle_deleted`` are the two
    cases the worker dispatches on, derived from
    :class:`MdChangeState.change_type`. Either may raise — the worker
    catches and classifies (``RecoverableError`` vs unrecoverable) to
    drive the retry / failed-state lifecycle.
    """

    def __init__(self, deps: HandlerDeps) -> None:
        self._deps = deps

    @abc.abstractmethod
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        """Reconcile LanceDB for an md path that exists on disk.

        Args:
            md_path: Path **relative** to the memory root (e.g.
                ``users/u1/episodes/episode-2026-05-14.md``).

        Returns:
            A :class:`HandlerOutcome` summarising the diff resolution.
        """

    @abc.abstractmethod
    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        """Wipe every LanceDB row that points back to ``md_path``.

        Idempotent: a path that was never indexed (or already wiped)
        returns an outcome with zero deletes; the worker still marks
        the state row ``done`` so the queue stays clean.
        """
