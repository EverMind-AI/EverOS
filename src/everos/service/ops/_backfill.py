"""Backfill's shared vocabulary: dependencies, presenter, results, summary.

Under soft-dependency (embed-optional), a user who starts on Tier 1 (LLM
only) accumulates memory rows with ``vector IS NULL``. Once they configure
embedding, three phases upgrade them, in order:

- ``vectors``  -- re-embed missing vectors on existing rows;
- ``clusters`` -- build clusters on the newly embedded episodes / cases;
- ``skills``   -- extract agent skills from the clustered cases.

Each phase is a private step module of the ops facade; this module holds
what they share. Presentation is the caller's: every phase reports progress
and asks for confirmation through a :class:`BackfillPresenter`, which the
CLI implements with typer and tests with a recording stub.
"""

from __future__ import annotations

__all__ = [
    "BACKFILL_LOGGER",
    "BACKFILL_PHASES",
    "BackfillDeps",
    "BackfillOutcome",
    "BackfillPresenter",
    "BackfillSummary",
    "CascadeSync",
    "ClustersPhaseResult",
    "NullBackfillPresenter",
    "OmeLockProbe",
    "PhaseResult",
    "ReplayEngine",
    "SkillFiles",
    "SkillsPhaseResult",
    "VectorsPhaseResult",
    "format_tokens",
    "progress_due",
]

import dataclasses
import enum
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.infra.persistence.index import IndexMaintenance, VectorBackfill
    from everos.memory.ports import (
        AgentSkillIndex,
        ClusterStore,
        Event,
        Scope,
        StrategySpec,
    )

BACKFILL_PHASES: tuple[str, ...] = ("vectors", "clusters", "skills")
"""The phases, in the order ``all`` runs them."""

BACKFILL_LOGGER = "everos.service.backfill"
"""The logger the phases log under, one name for operators to filter on.

The vectors phase's row-level events (``cascade_backfill_row_*``,
``cascade_backfill_table_*``) come from the index module that does the work,
:mod:`everos.infra.persistence.index.vector_backfill`.
"""


class ReplayEngine(Protocol):
    """A throw-away scheduler the cluster and skill phases replay events into."""

    async def start(self) -> None:
        """Start dispatching.

        Raises:
            MemoryRootLockedError: Another process (a server) holds the root.
        """
        ...

    async def stop(self) -> None: ...

    async def emit(self, event: Event) -> None: ...

    async def wait_idle(self, *, timeout: float) -> bool: ...  # noqa: ASYNC109


class SkillFiles(Protocol):
    """Whether a skill document already names a cluster (the markdown truth)."""

    async def names_cluster(self, scope: Scope, cluster_id: str) -> bool: ...


class CascadeSync(Protocol):
    """One scan + drain of the md -> index projection, optionally per kind."""

    async def sync_once(self, *, kinds: set[str] | None = None) -> int: ...


class OmeLockProbe(Protocol):
    """Whether no other process holds the memory root's exclusive lock."""

    def is_free(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class BackfillDeps:
    """What the backfill phases read, write and replay through.

    Assembled once by :func:`everos.service.build_application`.

    Attributes:
        embedding: Embedding provider; ``None`` blocks every phase.
        vectors: Finds and re-embeds the index rows the vectors phase
            upgrades.
        vector_tables: Every index table that carries a vector column, by
            table name -- the episodes / agent cases the clusters phase
            replays.
        clusters: The cluster store the replayed strategies write.
        skills: The agent-skill index the skills phase counts.
        skill_files: Whether a ``SKILL.md`` already names a cluster.
        engine_for: Builds an unstarted replay engine running only the
            given strategies.
        ensure_schema: Creates the system-DB tables when they do not exist.
        cascade: Projects the skill documents the skills phase writes.
        ome_lock: Probes the lock a running server holds.
        wait_seconds: How long a replay may take to drain before the phase
            reports what it has.
    """

    embedding: EmbeddingProvider | None
    vectors: VectorBackfill
    vector_tables: Mapping[str, IndexMaintenance]
    clusters: ClusterStore
    skills: AgentSkillIndex
    skill_files: SkillFiles
    engine_for: Callable[[Sequence[StrategySpec]], ReplayEngine]
    ensure_schema: Callable[[], Awaitable[object]]
    cascade: CascadeSync
    ome_lock: OmeLockProbe
    wait_seconds: float = 300.0


class BackfillPresenter(Protocol):
    """How the phases surface progress and ask for confirmation.

    Every method may be called from an async context. Sync methods must not
    block on meaningful I/O; :meth:`confirm` is async so an implementation
    can run a blocking prompt in a thread. :meth:`confirm` returns ``False``
    to decline the phase; ``auto_yes=True`` must short-circuit any prompt
    and return ``True``. An interrupt raised at a real prompt propagates to
    the caller of :meth:`OpsService.backfill` untouched.
    """

    def phase_started(self, phase: str) -> None:
        """A phase (one of :data:`BACKFILL_PHASES`) is about to run."""
        ...

    def nothing_to_backfill(self, message: str, *, scan_failed: bool = False) -> None:
        """Report that a phase found nothing to do.

        ``scan_failed`` tells "the input space is empty" (``False``) from
        "rows were there but the scan dropped some to storage errors"
        (``True``); presentation keys off the flag, never the wording.
        """
        ...

    def capability_missing(
        self, *, provider: str, feature: str, message: str
    ) -> None: ...
    def server_running(self) -> None: ...
    def estimate_vectors(self, rows: int, tokens: int) -> None: ...
    def estimate_clusters(self, episodes: int, cases: int) -> None: ...
    def estimate_skills(self, cases: int, clusters: int) -> None: ...
    async def confirm(self, prompt: str, *, auto_yes: bool) -> bool: ...
    def emit_progress(self, done: int, total: int) -> None: ...
    def row_progress(self, table_name: str, done: int, total: int) -> None: ...
    def phase_1_complete(self, rows_processed: int, rows_failed: int) -> None: ...
    def phase_2_complete(self, before: int, after: int, emitted: int) -> None: ...
    def phase_3_complete(
        self, skills_before: int, skills_after: int, emitted: int
    ) -> None: ...


class NullBackfillPresenter:
    """A :class:`BackfillPresenter` that shows nothing and confirms everything."""

    def phase_started(self, phase: str) -> None:
        return None

    def nothing_to_backfill(self, message: str, *, scan_failed: bool = False) -> None:
        return None

    def capability_missing(self, *, provider: str, feature: str, message: str) -> None:
        return None

    def server_running(self) -> None:
        return None

    def estimate_vectors(self, rows: int, tokens: int) -> None:
        return None

    def estimate_clusters(self, episodes: int, cases: int) -> None:
        return None

    def estimate_skills(self, cases: int, clusters: int) -> None:
        return None

    async def confirm(self, prompt: str, *, auto_yes: bool) -> bool:
        return True

    def emit_progress(self, done: int, total: int) -> None:
        return None

    def row_progress(self, table_name: str, done: int, total: int) -> None:
        return None

    def phase_1_complete(self, rows_processed: int, rows_failed: int) -> None:
        return None

    def phase_2_complete(self, before: int, after: int, emitted: int) -> None:
        return None

    def phase_3_complete(
        self, skills_before: int, skills_after: int, emitted: int
    ) -> None:
        return None


@dataclasses.dataclass
class VectorsPhaseResult:
    """The vectors phase's outcome.

    ``aborted`` tells "the user declined" from "nothing to backfill"; both
    otherwise leave every counter at zero. ``blocked_by_capability`` names
    the missing provider; ``blocked_by_server`` is set when the lock probe
    found a running server, before any embedding call was paid for.
    """

    rows_processed: int = 0
    rows_failed: int = 0
    tokens_embedded: int = 0
    aborted: bool = False
    blocked_by_capability: str | None = None
    blocked_by_server: bool = False


@dataclasses.dataclass
class ClustersPhaseResult:
    """The clusters phase's outcome.

    ``events_emitted`` counts every replayed row (real emits plus rows a
    previous run already clustered). ``clusters_before`` / ``clusters_after``
    are the total cluster counts around the replay; ``drained`` is ``False``
    when the replay did not finish within the wait.
    """

    events_emitted: int = 0
    clusters_before: int = 0
    clusters_after: int = 0
    drained: bool = True
    aborted: bool = False
    blocked_by_server: bool = False
    blocked_by_capability: str | None = None


@dataclasses.dataclass
class SkillsPhaseResult:
    """The skills phase's outcome.

    ``events_emitted`` counts the replayed (case, cluster) pairs;
    ``skills_before`` / ``skills_after`` are index counts taken after the
    follow-up cascade sync, so growth is already searchable.
    """

    events_emitted: int = 0
    skills_before: int = 0
    skills_after: int = 0
    drained: bool = True
    aborted: bool = False
    blocked_by_server: bool = False
    blocked_by_capability: str | None = None


PhaseResult = VectorsPhaseResult | ClustersPhaseResult | SkillsPhaseResult
"""What one phase returns."""


class BackfillOutcome(enum.StrEnum):
    """How a backfill run ended."""

    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    """Every phase ran, but some rows failed embedding even row by row."""
    ABORTED = "aborted"
    BLOCKED_BY_CAPABILITY = "blocked_by_capability"
    BLOCKED_BY_SERVER = "blocked_by_server"


@dataclasses.dataclass
class BackfillSummary:
    """Each phase's result, filled in as the phases finish.

    A field stays ``None`` for a phase that did not run, so a summary shows
    exactly the phases that executed.
    """

    vectors: VectorsPhaseResult | None = None
    clusters: ClustersPhaseResult | None = None
    skills: SkillsPhaseResult | None = None

    @property
    def rows_failed(self) -> int:
        """Rows that failed embedding; only the vectors phase counts rows."""
        return self.vectors.rows_failed if self.vectors else 0

    def record(self, phase: str, result: PhaseResult) -> None:
        setattr(self, phase, result)


def format_tokens(n: int) -> str:
    """A token count in K / M shorthand -- never a price, which is per provider."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n} tokens"


def progress_due(done: int, total: int, *, every: int = 100) -> bool:
    """Whether to report replay progress now: every ``every`` rows and at the end."""
    return done % every == 0 or done == total
