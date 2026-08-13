"""Backfill phase runners for ``everos cascade backfill``.

Under soft-dependency (embed-optional), a user who starts on Tier 1
(LLM only) accumulates memory rows with ``vector IS NULL``. Once they
configure embedding and move to Tier 2/3, those rows stay keyword-only
until backfilled. The three phase runners here upgrade them:

- Phase 1 (:func:`_run_phase_vectors`)  — re-embed missing vectors on
  existing rows.
- Phase 2 (:func:`_run_phase_clusters`) — build clusters on the newly-
  embedded episodes / agent cases.
- Phase 3 (:func:`_run_phase_skills`)   — extract agent skills from
  clustered cases.

Presentation is not the memory layer's job. Every phase runner takes a
:class:`BackfillPresenter` kwarg — the CLI (``entrypoints/cli/commands/
_backfill_cmd.py``) supplies a typer-backed implementation; tests use a
no-op / recording stub. This is the sole seam through which the
otherwise-pure phase logic surfaces progress and confirmation to the
caller. Fixes PR #361 review finding M11: ``memory`` used to import
``typer`` + ``click`` directly and silently bypass the architecture
layering rule (import-linter does not check third-party imports).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import anyio
import anyio.to_thread
import portalocker

from everos.component.embedding import EmbeddingProvider, get_embedding_capability
from everos.component.tokenizer import build_tokenizer
from everos.component.utils.datetime import to_timestamp_ms
from everos.core.errors import ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.core.persistence import MarkdownReader, MemoryRoot, SQLModel
from everos.core.persistence.lancedb import BaseLanceTable, LanceRepoBase
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.infra.ome.exceptions import EngineLockHeldError
from everos.infra.persistence.lancedb import (
    BUSINESS_SCHEMAS_WITH_VECTOR,
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    agent_case_repo,
    agent_skill_repo,
    atomic_fact_repo,
    episode_repo,
    foresight_repo,
    get_table,
    knowledge_topic_repo,
)
from everos.infra.persistence.markdown import AgentSkillFrontmatter
from everos.infra.persistence.sqlite import cluster_repo, get_engine
from everos.memory.cascade.worker import (
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
)
from everos.memory.events import (
    AgentCaseExtracted,
    EpisodeExtracted,
    SkillClusterUpdated,
)
from everos.memory.strategies import (
    extract_agent_skill,
    trigger_profile_clustering,
    trigger_skill_clustering,
)

from .orchestrator import CascadeOrchestrator

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class BackfillPhase:
    """One backfill phase: identity plus the copy shown at the prompt.

    ``slug`` matches the CLI ``--phase`` value one-to-one (``vectors`` /
    ``clusters`` / ``skills``), so phase selection is plain string
    equality with no translation layer. The concrete :data:`PHASES`
    tuple carrying user-facing ``title`` / ``detail`` strings lives in
    ``entrypoints/cli/commands/_backfill_cmd.py`` because those strings
    are CLI copy, not domain data.
    """

    number: int
    slug: str
    title: str
    detail: str


class BackfillPresenter(Protocol):
    """Callback interface phase runners use to surface progress and
    confirmation to the caller.

    The CLI (``entrypoints/cli/commands/_backfill_cmd.py``) supplies a
    typer-backed implementation (:class:`TyperPresenter`); tests inject
    a no-op / recording stub. Kept as a :class:`~typing.Protocol` (not
    an ABC) so structural typing lets any object with matching methods
    fit without explicit inheritance.

    Contract:

    - Every method may be called from an async context. Sync methods
      must not block for meaningful I/O; :meth:`confirm` is async so an
      implementation can dispatch the actual (blocking) prompt to a
      thread.
    - :meth:`confirm` returns ``True`` to proceed, ``False`` to abort
      the phase with exit code 1. Ctrl-C at a real prompt is expected
      to surface as :class:`click.exceptions.Abort` (a
      :class:`RuntimeError` subclass) — the orchestrator in
      ``entrypoints/cli/commands/_backfill_cmd.py`` catches it and maps
      to exit 130. Memory-layer runners must not catch it here; the
      abort is a caller concern.
    - ``auto_yes=True`` on :meth:`confirm` MUST short-circuit any
      interactive prompt and return ``True``.
    """

    def nothing_to_backfill(self, message: str, *, scan_failed: bool = False) -> None:
        """Report a phase yielded nothing to do.

        ``scan_failed`` distinguishes "phase had no data because the
        input space is empty" (``False`` — green happy path) from
        "phase had rows but the scan itself dropped some to storage
        errors" (``True`` — yellow degradation warning). The CLI layer
        colour-picks off this flag; **do not sniff substrings of
        ``message``** (fragile — a domain-side wording change would
        silently flip the colour). ``message`` remains a plain string
        so implementations can present it verbatim.
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
    """No-op :class:`BackfillPresenter` for callers that don't need output.

    Kept in the memory layer so tests and non-CLI invocations (offline
    scripts, embedded harnesses) can construct one without importing
    ``entrypoints``. :meth:`confirm` honours ``auto_yes=True`` and
    otherwise returns ``True`` — the phase runners still enforce
    correctness gates through their return values, not through the
    prompt.
    """

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


# ── Phase 1 (vectors) — table wiring ─────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _TableSpec:
    """One business table's Phase 1 wiring.

    ``text_of`` pulls the same source text each cascade handler feeds to
    the embedder out of a raw LanceDB row dict (see ``memory/cascade/
    handlers/*.py``). ``subject_of`` is only set for ``episode``, whose
    ``subject_vector`` is a second, independent embed of the row's
    ``subject`` column — mirrors ``EpisodeHandler._build_row``.
    """

    schema: type[BaseLanceTable]
    repo: LanceRepoBase[Any]
    text_of: Callable[[dict[str, Any]], str]
    subject_of: Callable[[dict[str, Any]], str | None] | None = None


def _agent_skill_embed_text(row: dict[str, Any]) -> str:
    """Mirrors ``AgentSkillHandler``: embed source is name + description."""
    return "\n".join(s for s in (row.get("name"), row.get("description")) if s)


_TABLE_SPECS: tuple[_TableSpec, ...] = (
    _TableSpec(
        Episode,
        episode_repo,
        lambda r: r["episode"],
        subject_of=lambda r: r.get("subject") or None,
    ),
    _TableSpec(AtomicFact, atomic_fact_repo, lambda r: r["fact"]),
    _TableSpec(Foresight, foresight_repo, lambda r: r["foresight"]),
    _TableSpec(AgentCase, agent_case_repo, lambda r: r["task_intent"]),
    _TableSpec(AgentSkill, agent_skill_repo, _agent_skill_embed_text),
    _TableSpec(KnowledgeTopic, knowledge_topic_repo, lambda r: r["summary"]),
)

# Load-bearing invariant: ``_TABLE_SPECS`` must cover exactly the same
# tables as :data:`BUSINESS_SCHEMAS_WITH_VECTOR`. The infra list is the
# public source of truth (consumed by the LanceDB lifespan's unbackfilled
# hint AND by ``migrate_table_schemas``). Adding a new business table
# there without adding a matching ``_TableSpec`` here silently drops it
# from backfill — the hint would count rows the backfill CLI never
# touches. Fail loud at import time so any drift shows up before a test
# even starts.
_TABLE_SPEC_NAMES = {spec.schema.TABLE_NAME for spec in _TABLE_SPECS}
_BUSINESS_SCHEMA_NAMES = {s.TABLE_NAME for s in BUSINESS_SCHEMAS_WITH_VECTOR}
if _TABLE_SPEC_NAMES != _BUSINESS_SCHEMA_NAMES:
    raise RuntimeError(
        f"_TABLE_SPECS drift: BUSINESS_SCHEMAS_WITH_VECTOR has "
        f"{_BUSINESS_SCHEMA_NAMES}, _TABLE_SPECS covers "
        f"{_TABLE_SPEC_NAMES}. Update _TABLE_SPECS to include every "
        f"business table with a nullable vector column."
    )

_EMBED_BATCH_SIZE = 32
"""Rows grouped per ``embed_batch`` call (one for the primary text, one
more only if the group has subject rows) — not the provider's own
request chunk size. The provider's configured ``batch_size`` /
``max_concurrent`` (``embedding.batch_size`` / ``embedding.max_concurrent``
in settings) govern the actual in-flight request shape from there; this
just bounds per-batch failure isolation + progress feedback granularity
for a bulk migration."""


@dataclasses.dataclass(frozen=True)
class _NullVectorRow:
    """One row missing (at least one side of) its vector.

    ``tokens`` is the real token count (via the same tokenizer cascade
    uses for BM25), not an estimate — computed once here and never
    recomputed before the embed call.

    ``needs_primary`` / ``needs_subject`` mark which side(s) of the
    row's embedding are still NULL. Round-1 assumed "in the backlog ⇒
    primary vector is NULL"; round-2 widened the scan filter to also
    pick up rows where only ``subject_vector`` is NULL (Episode's
    orthogonal second embed can fail on its own — see
    :func:`_null_filter`). ``_backfill_table`` reads these flags to
    embed only the side that actually needs work, so a re-run after a
    partial failure doesn't waste an ``embed_batch`` call on the side
    that already succeeded.

    Both default to ``True`` so pre-existing test fixtures that build
    :class:`_NullVectorRow` directly keep their round-1 semantics
    (embed everything).
    """

    id: str
    text: str
    subject_text: str | None
    tokens: int
    needs_primary: bool = True
    needs_subject: bool = True


@dataclasses.dataclass(frozen=True)
class _TableBacklog:
    spec: _TableSpec
    rows: list[_NullVectorRow]

    @property
    def table_name(self) -> str:
        return self.spec.schema.TABLE_NAME


@dataclasses.dataclass
class _PhaseResult:
    """Phase 1 outcome.

    ``aborted`` distinguishes "user declined the confirmation prompt"
    (caller exits 1) from "there was nothing to backfill" (caller exits
    0) — both otherwise leave every counter at zero.
    ``blocked_by_capability`` names the missing provider (currently only
    ``"embedding"``) when a preflight short-circuits the phase before it
    can start; the CLI orchestrator maps that to exit code 2.
    ``blocked_by_server`` mirrors the Phase 2 / Phase 3 field — set when
    the OME lock preflight fails (server running), so ``run_backfill``
    can exit 3 before any embed API call is made.
    """

    rows_processed: int = 0
    rows_failed: int = 0
    tokens_embedded: int = 0
    aborted: bool = False
    blocked_by_capability: str | None = None
    blocked_by_server: bool = False


def _q(value: str) -> str:
    """Defensive SQL-quote escape (mirrors the lancedb repo convention)."""
    return value.replace("'", "''")


def _null_filter(spec: _TableSpec) -> str:
    """Where-clause the scan uses to spot rows that need (some) embedding.

    Episode carries a ``subject_vector`` alongside ``vector`` — the two
    are orthogonal embeds (see :class:`Episode`), and either being NULL
    means the row is incompletely embedded. Round-1 filtered only on
    ``vector IS NULL``, which silently missed rows where the primary
    embed succeeded but ``_embed_subject_batch`` failed on its own
    (``subject_vector`` left NULL). Round-2 widens the filter to
    ``vector IS NULL OR subject_vector IS NULL`` so those rows
    re-enter the backlog on the next scan.

    Other tables have no subject column; adding the OR clause there
    would reference an unknown field and LanceDB would reject the
    query, so they keep the round-1 filter.
    """
    if spec.schema.TABLE_NAME == Episode.TABLE_NAME:
        return "vector IS NULL OR subject_vector IS NULL"
    return "vector IS NULL"


class _RowSkipped:
    """Sentinel returned by :func:`_extract_row` when the row shape is
    fine but nothing actually needs embedding (widened Episode filter
    matched on a legitimately-NULL ``subject_vector``). Distinct from
    ``None`` — which the caller tallies as a scan-failure — so the
    two paths never conflate."""


_ROW_SKIPPED = _RowSkipped()


def _extract_row(
    raw: dict[str, Any], spec: _TableSpec, tokenizer: Any
) -> _NullVectorRow | _RowSkipped | None:
    """Pull one raw LanceDB row's embed-source text, token count, and
    which side(s) still need embedding.

    Return values:

    - :class:`_NullVectorRow` — the row has real work; carries the
      ``needs_primary`` / ``needs_subject`` flags read off the raw
      vector columns.
    - :data:`_ROW_SKIPPED` — the row's shape is fine but nothing needs
      embedding (widened Episode filter matched on a legitimately-NULL
      ``subject_vector``). Silent; caller drops it.
    - ``None`` — the row's shape is broken (schema drift / bad row).
      Logged; caller tallies as scan-failed.
    """
    try:
        text = spec.text_of(raw)
        subject_text = spec.subject_of(raw) if spec.subject_of else None
        row_id = raw["id"]
    except (KeyError, TypeError) as exc:
        logger.warning(
            "cascade_backfill_scan_extract_failed",
            table=spec.schema.TABLE_NAME,
            row_id=raw.get("id"),
            error=repr(exc),
        )
        return None
    needs_primary = raw.get("vector") is None
    # Subject-side is only meaningful when the spec has a subject
    # projection AND this row actually carries subject text. A NULL
    # ``subject_vector`` on a row whose subject was legitimately
    # absent is not something to fix — it will always be NULL.
    needs_subject = (
        spec.subject_of is not None
        and subject_text is not None
        and raw.get("subject_vector") is None
    )
    if not needs_primary and not needs_subject:
        return _ROW_SKIPPED
    token_source = f"{text} {subject_text}" if needs_subject else text
    return _NullVectorRow(
        id=row_id,
        text=text,
        subject_text=subject_text,
        tokens=len(tokenizer.tokenize(token_source)),
        needs_primary=needs_primary,
        needs_subject=needs_subject,
    )


async def _scan_null_vector_backlog() -> tuple[list[_TableBacklog], int]:
    """Fetch every row across the 6 business tables that still needs
    (some) embedding.

    One query per table, tokenising each row's embed-source text on the
    way so the pre-run estimate and the later embed call read off the
    exact same text. Returns the backlog plus a count of rows whose
    extraction failed (bad row shape) — tallied by the caller into the
    phase's ``rows_failed``, never raised.

    Round-2 changes:

    - The Episode filter is widened to
      ``vector IS NULL OR subject_vector IS NULL`` via
      :func:`_null_filter` so a subject-only failure doesn't hide the
      row from subsequent backfill runs.
    - The prior ``count_rows`` + ``.limit(null_count)`` split is
      dropped. Under concurrent server writes (Phase 1 is documented
      as concurrent-safe), a fresh NULL-vector row inserted between
      the count and the query would push a legitimate row past the
      cap. LanceDB's ``.to_list()`` streams the whole predicate result;
      the ``where`` filter bounds it naturally.
    """
    tokenizer = build_tokenizer()
    backlog: list[_TableBacklog] = []
    scan_failed = 0
    for spec in _TABLE_SPECS:
        table = await get_table(spec.schema.TABLE_NAME, spec.schema)
        raw_rows = await table.query().where(_null_filter(spec)).to_list()
        rows: list[_NullVectorRow] = []
        for raw in raw_rows:
            row = _extract_row(raw, spec, tokenizer)
            if row is None:
                scan_failed += 1
                continue
            if isinstance(row, _RowSkipped):
                continue
            rows.append(row)
        backlog.append(_TableBacklog(spec, rows))
    return backlog, scan_failed


def _truncated_text_prefix(text: str, *, limit: int = 100) -> str:
    """Return the first ``limit`` chars of ``text``, with an ellipsis when
    truncated. Used for per-row failure logs — never log full text, it
    may be user data / PII."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def _embed_primary_batch(
    provider: EmbeddingProvider, batch: list[_NullVectorRow], table_name: str
) -> list[list[float] | None]:
    """Embed the batch's primary text with per-row fallback.

    First attempts the batched ``provider.embed_batch(...)`` call — the
    happy path. On any exception (network glitch, one poison row whose
    text exceeds the provider's context limit and 400s the whole batch,
    etc.), logs the batch failure and falls back to per-row
    ``provider.embed(...)`` calls. Each per-row failure is logged with
    the row's id and a truncated text prefix (100 chars) so operators
    can identify poison rows and quarantine them.

    Returns a same-length list aligned with the input ``batch``: each
    entry is either the row's vector, or ``None`` for a row that failed
    even the per-row retry. The caller writes only rows with non-None
    vectors — failed rows keep ``vector IS NULL`` on disk and re-enter
    the next scan's NULL-vector backlog.
    """
    try:
        vectors = await provider.embed_batch([row.text for row in batch])
        return list(vectors)
    except Exception as exc:
        logger.warning(
            "cascade_backfill_batch_embed_failed_falling_back_per_row",
            table=table_name,
            batch_size=len(batch),
            error=repr(exc),
        )
    # Per-row fallback: isolate the poison row(s) so the rest advance.
    results: list[list[float] | None] = []
    for row in batch:
        try:
            vec = await provider.embed(row.text)
        except Exception as exc:
            logger.warning(
                "cascade_backfill_row_embed_failed",
                table=table_name,
                row_id=row.id,
                text_prefix=_truncated_text_prefix(row.text),
                error=repr(exc),
            )
            results.append(None)
            continue
        results.append(vec)
    return results


async def _embed_subject_batch(
    provider: EmbeddingProvider, batch: list[_NullVectorRow], table_name: str
) -> dict[str, list[float]]:
    """Embed the batch's ``subject_text`` with per-row fallback (episode's
    second, independent embed — see ``EpisodeHandler._build_row``).

    A separate batch call because not every row has a subject; folding
    it into the primary call would either skip absent subjects (index
    mismatch) or force a dense-but-wrong text list.

    Returns an ``{id: vector}`` map. Rows whose per-row retry also failed
    are absent from the map, and their ``subject_vector`` stays NULL —
    the widened ``vector IS NULL OR subject_vector IS NULL`` scan filter
    (M1) picks them up on the next backfill run.
    """
    subject_rows = [row for row in batch if row.subject_text is not None]
    if not subject_rows:
        return {}
    try:
        vectors = await provider.embed_batch(
            [row.subject_text for row in subject_rows]  # type: ignore[misc]
        )
        return {row.id: vec for row, vec in zip(subject_rows, vectors, strict=True)}
    except Exception as exc:
        logger.warning(
            "cascade_backfill_subject_batch_embed_failed_falling_back_per_row",
            table=table_name,
            batch_size=len(subject_rows),
            error=repr(exc),
        )
    results: dict[str, list[float]] = {}
    for row in subject_rows:
        text = row.subject_text
        assert text is not None  # guarded by the filter above
        try:
            vec = await provider.embed(text)
        except Exception as exc:
            logger.warning(
                "cascade_backfill_row_subject_embed_failed",
                table=table_name,
                row_id=row.id,
                text_prefix=_truncated_text_prefix(text),
                error=repr(exc),
            )
            continue
        results[row.id] = vec
    return results


async def _backfill_table(
    backlog: _TableBacklog,
    provider: EmbeddingProvider,
    *,
    presenter: BackfillPresenter,
) -> _PhaseResult:
    """Embed + write back every row in one table's backlog, in batches.

    Each batch is split by which side each row still needs — the
    orthogonal side (Episode's ``subject_vector``) is a separate
    ``embed_batch`` call, and a row that already has one side filled
    (e.g. primary succeeded on a prior run, subject failed) only pays
    the cost of the missing side. When both sides have work in the
    same batch the two calls run concurrently under ``asyncio.gather``
    (round-1 concurrency invariant preserved).

    Failure isolation:

    - Whole primary batch failing → those rows keep ``vector IS NULL``
      on disk; the widened scan filter picks them up on the next run.
      Counted as ``rows_failed`` on this pass.
    - Whole subject batch failing → the touched rows keep
      ``subject_vector IS NULL`` on disk; the widened scan filter
      picks them up on the next run (round-1 fix #10's silent
      corruption is closed here).
    - Per-row write failing → tally the row as ``rows_failed`` and
      leave it for the next scan.

    Write-back stays per-row: LanceDB's partial-column ``update``
    takes a single-row predicate, so a failed write only fails that
    one row.
    """
    result = _PhaseResult()
    rows = backlog.rows
    for start in range(0, len(rows), _EMBED_BATCH_SIZE):
        batch = rows[start : start + _EMBED_BATCH_SIZE]
        primary_rows = [r for r in batch if r.needs_primary]
        subject_rows = [
            r for r in batch if r.needs_subject and r.subject_text is not None
        ]

        # Concurrent embed only for the sides that have work this
        # batch. ``asyncio.gather`` under both-need shape preserves the
        # round-1 wall-clock invariant; one-sided batches skip the
        # gather to avoid a wasted ``embed_batch({})`` round-trip.
        primary_vectors: list[list[float] | None]
        subject_vectors: dict[str, list[float]]
        if primary_rows and subject_rows:
            primary_vectors, subject_vectors = await asyncio.gather(
                _embed_primary_batch(provider, primary_rows, backlog.table_name),
                _embed_subject_batch(provider, subject_rows, backlog.table_name),
            )
        elif primary_rows:
            primary_vectors = await _embed_primary_batch(
                provider, primary_rows, backlog.table_name
            )
            subject_vectors = {}
        elif subject_rows:
            primary_vectors = []
            subject_vectors = await _embed_subject_batch(
                provider, subject_rows, backlog.table_name
            )
        else:
            # Purely defensive: the scan filters no-op rows out via
            # ``_ROW_SKIPPED``, so an empty split shouldn't happen in
            # practice. Still emit progress so the readout stays
            # monotonic.
            done = min(start + _EMBED_BATCH_SIZE, len(rows))
            presenter.row_progress(backlog.table_name, done, len(rows))
            continue

        # ``primary_vectors`` is a same-length list aligned with
        # ``primary_rows``: successful rows carry the vector, poison
        # rows (whose per-row retry also failed) carry ``None``. Count
        # the poison rows as ``rows_failed`` — they keep ``vector IS
        # NULL`` on disk and re-enter the next scan.
        primary_map: dict[str, list[float]] = {}
        for r, v in zip(primary_rows, primary_vectors, strict=True):
            if v is None:
                result.rows_failed += 1
            else:
                primary_map[r.id] = v
        # ``subject_vectors`` is already an ``{id: vec}`` dict; ids
        # absent mean the per-row subject retry also failed and the
        # row keeps ``subject_vector IS NULL``, re-entering the backlog
        # via the widened NULL scan next run.

        for row in batch:
            updates: dict[str, Any] = {}
            if row.id in primary_map:
                updates["vector"] = primary_map[row.id]
            if row.id in subject_vectors:
                updates["subject_vector"] = subject_vectors[row.id]
            if not updates:
                continue
            # A row is only "fully processed" if every side it needed
            # actually made it into ``updates``. When the primary
            # succeeded but the subject batch failed (or per-row retry
            # dropped the subject), the row still carries a NULL side
            # on disk and the widened scan filter (``vector IS NULL OR
            # subject_vector IS NULL``) re-picks it next run —
            # automation must see a non-zero ``rows_failed`` for the
            # exit code to escalate to COMPLETED_WITH_FAILURES (4)
            # instead of falsely reporting SUCCESS (0). Two guards
            # tighten the check: (1) consult per-side ``needs_*`` so a
            # row where ``needs_primary=False`` (primary already on
            # disk from a prior pass) isn't counted as a gap; (2) gate
            # the subject-side arm on the SPEC having a subject_of
            # extractor at all — tables without a subject column
            # (e.g. atomic_fact) intentionally leave every row absent
            # from ``subject_vectors`` and must not be flagged.
            spec_has_subject = backlog.spec.subject_of is not None
            row_has_subject = row.subject_text is not None
            side_gap = (row.needs_primary and row.id not in primary_map) or (
                spec_has_subject
                and row.needs_subject
                and row_has_subject
                and row.id not in subject_vectors
            )
            try:
                await backlog.spec.repo.update(updates, where=f"id = '{_q(row.id)}'")
            except Exception:
                result.rows_failed += 1
                logger.warning(
                    "cascade_backfill_row_write_failed",
                    table=backlog.table_name,
                    row_id=row.id,
                    exc_info=True,
                )
                continue
            # Partial-side gap: the write succeeded for at least one
            # side but some needed side is still NULL. Count against
            # ``rows_failed`` so the CLI's exit-4 (COMPLETED_WITH_FAILURES)
            # semantics fires instead of silently exit-0.
            if side_gap:
                result.rows_failed += 1
                logger.warning(
                    "cascade_backfill_row_partial_embed",
                    table=backlog.table_name,
                    row_id=row.id,
                    missing_side=(
                        "primary" if row.id not in primary_map else "subject"
                    ),
                )
                continue
            # A row is "processed" once every side it needed advanced
            # this pass.
            result.rows_processed += 1
            result.tokens_embedded += row.tokens

        done = min(start + _EMBED_BATCH_SIZE, len(rows))
        presenter.row_progress(backlog.table_name, done, len(rows))

    # Compact fragments accumulated during backfill. Every per-row
    # update opens a fresh LanceDB manifest version; without periodic
    # optimize() the on-disk directory grows unbounded (same failure
    # mode as v1.1.3's FTS optimize gap, #336 /
    # lance-format/lance#7653). Gate on rows_processed > 0 so a no-op
    # backlog doesn't pay the compact cost. Best-effort maintenance:
    # a failure here does not invalidate the writes.
    #
    # ``cleanup_older_than=timedelta(0)`` also physically prunes
    # older manifest versions right now — without it, ``optimize()``
    # only compacts fragments and leaves the pre-compaction manifest
    # chain on disk, so the ``.index/lancedb`` directory still grows
    # (round-4 review M2). Backfill has no reason to preserve older
    # manifest versions — pruning immediately mirrors the migration
    # path (see ``lancedb/__init__.py:140``).
    if result.rows_processed > 0:
        try:
            # optimize() compacts the per-row-update fragments; prune()
            # physically reclaims the superseded manifest versions. Split
            # after the repo API separated them (compact is lock-free, prune
            # runs under the write lock) and cross-process safe because prune
            # passes delete_unverified=False.
            #
            # Keep the daemon's retention window rather than reclaiming at
            # zero age: the window's job is to outlive an in-flight read (a
            # /search holding a version reference), and this runs in a
            # separate process where the write lock cannot fence one. Files
            # younger than the window are reclaimed by the next daemon prune.
            await backlog.spec.repo.optimize()
            await backlog.spec.repo.prune(
                dt.timedelta(seconds=DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS)
            )
            logger.info(
                "cascade_backfill_table_optimized",
                table=backlog.table_name,
                rows_processed=result.rows_processed,
            )
        except Exception as exc:
            logger.warning(
                "cascade_backfill_table_optimize_failed",
                table=backlog.table_name,
                error=repr(exc),
            )
    return result


async def _run_phase_vectors(
    *, auto_yes: bool, presenter: BackfillPresenter
) -> _PhaseResult:
    """Re-embed rows with a NULL primary vector (and, for Episode, a
    NULL subject vector) across the 6 business tables.

    Preflight capability → scan + tokenize → surface the estimate via
    ``presenter`` → confirm once → embed in batches, writing each row
    back via ``LanceRepoBase.update`` (a partial-column update, not a
    full-row ``merge_insert`` — Phase 1 only ever changes ``vector`` /
    ``subject_vector``). Per-row failures (including scan-time
    extraction failures) are tallied and logged; the phase runs to
    completion regardless.

    Preflight runs FIRST — before scan + tokenize + confirm — so a
    user with a large accumulated backlog and no embedding configured
    exits fast with a toml-hint instead of paying an O(N) scan they
    can't act on. The empty-backlog path also preflights, so a fresh
    install still gets the hint rather than a bare "nothing to
    backfill" green message.

    After each table's backfill loop, :func:`_backfill_table` calls
    ``optimize()`` to compact the manifest / fragment accumulation
    from per-row updates. Same treatment as v1.1.3's FTS index
    optimize (#336) — LanceDB per-row writes open fragments that would
    otherwise grow unbounded.
    """
    # ── preflight first ────────────────────────────────────────────
    # Only ``.available`` is inspected here; the actual provider is
    # resolved after the confirm prompt so a declined y/N never
    # touches ``require()``.
    capability = get_embedding_capability()
    if not capability.available:
        err = ProviderNotConfiguredError(provider="embedding", feature="backfill")
        presenter.capability_missing(
            provider="embedding", feature="backfill", message=str(err)
        )
        return _PhaseResult(blocked_by_capability="embedding")

    # Phase 2 / Phase 3 both preflight the OME lock; Phase 1 must do
    # the same. Otherwise ``--phase all`` starts against a running
    # server, burns through Phase 1's embed API calls (real token
    # cost + potential commit conflict with the live cascade worker),
    # then only halts at Phase 2's preflight with exit 3. Fail fast
    # here so the user's next action is "stop the server" — not
    # "reimburse this month's embedding bill".
    if not await anyio.to_thread.run_sync(_probe_ome_lock_available):
        presenter.server_running()
        return _PhaseResult(aborted=True, blocked_by_server=True)

    backlog, scan_failed = await _scan_null_vector_backlog()
    total_rows = sum(len(tb.rows) for tb in backlog)
    if total_rows == 0:
        if scan_failed:
            presenter.nothing_to_backfill(
                f"Nothing to backfill — {scan_failed:,} row(s) could not be "
                "read and were skipped (see logs).",
                scan_failed=True,
            )
        else:
            presenter.nothing_to_backfill(
                "Nothing to backfill. All rows already have vectors."
            )
        return _PhaseResult(rows_failed=scan_failed)

    total_tokens = sum(row.tokens for tb in backlog for row in tb.rows)
    presenter.estimate_vectors(total_rows, total_tokens)
    if not await presenter.confirm(
        f"proceed with {total_rows:,} rows / ~{_format_tokens(total_tokens)} tokens",
        auto_yes=auto_yes,
    ):
        return _PhaseResult(aborted=True)

    # Resolve the provider only now — a declined y/N above should not
    # touch ``capability.require()`` (tests inspect that boundary).
    provider = capability.require()

    result = _PhaseResult(rows_failed=scan_failed)
    for tb in backlog:
        if not tb.rows:
            continue
        table_result = await _backfill_table(tb, provider, presenter=presenter)
        result.rows_processed += table_result.rows_processed
        result.rows_failed += table_result.rows_failed
        result.tokens_embedded += table_result.tokens_embedded

    presenter.phase_1_complete(result.rows_processed, result.rows_failed)
    return result


def _format_tokens(n: int) -> str:
    """Render a token count in K/M shorthand (provider per-M-token pricing
    convention) — never a currency estimate; see the embed-soft-dependency
    design doc §10 cost policy. Exposed as a module-level helper because
    both the memory-layer confirm-prompt string and the entrypoints
    summary reader consume the same format."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n} tokens"


# ── Phase 2 (clusters) — synthetic-event replay ──────────────────────────────


_CLUSTER_WAIT_TIMEOUT_SECONDS = 300.0
"""Cap on waiting for the ephemeral engine to drain (see
:meth:`OfflineEngine.wait_idle`). A backfill that never converges must not
hang the CLI forever; on timeout the phase still reports whatever cluster
count it observes and logs a warning rather than raising."""

_PROGRESS_EVERY = 100
"""Emit-progress cadence for the presenter — one call per row would flood
the presenter's output for a large backlog; a per-event log line would
flood the logs."""


@dataclasses.dataclass
class _ClusterPhaseResult:
    """Phase 2 outcome.

    ``events_emitted`` counts every synthesized ``EpisodeExtracted`` +
    ``AgentCaseExtracted`` event fanned into the ephemeral engine.
    ``clusters_before`` / ``clusters_after`` are the total ``cluster``
    row count (:meth:`_ClusterRepo.count`, across every owner/kind)
    taken immediately before dispatch and after the engine drains, so
    the caller can report growth. ``drained`` is ``False`` only if
    :meth:`OfflineEngine.wait_idle` hit :data:`_CLUSTER_WAIT_TIMEOUT_SECONDS`
    with runs still in flight. ``aborted`` mirrors :class:`_PhaseResult`.
    """

    events_emitted: int = 0
    clusters_before: int = 0
    clusters_after: int = 0
    drained: bool = True
    aborted: bool = False
    blocked_by_server: bool = False
    blocked_by_capability: str | None = None


async def _scan_all_rows(schema: type[BaseLanceTable]) -> list[dict[str, Any]]:
    """Fetch every row of ``schema``'s table, no filter.

    Phase 2 doesn't care whether a row already carries a vector — the
    cluster strategies re-embed the row's text themselves (see
    ``trigger_profile_clustering`` / ``trigger_skill_clustering``); it
    just needs every existing episode / agent case so it can
    synthesize the trigger event Tier 1's gated-off strategies never
    emitted (embed-requiring strategies are body-guarded off when
    ``get_embedding_capability().available`` is false — see
    :mod:`everos.memory.strategies`).

    Callers that need a ``parent_type`` filter (e.g. the Episode scan in
    :func:`_run_phase_clusters`, which excludes Reflection-merged rows)
    apply it themselves on the returned rows — this helper stays a plain
    unfiltered fetch shared by every business table.
    """
    table = await get_table(schema.TABLE_NAME, schema)
    total = await table.count_rows()
    if total == 0:
        return []
    return await table.query().limit(total).to_list()


def _episode_row_to_event(raw: dict[str, Any]) -> EpisodeExtracted:
    """Synthesize the ``EpisodeExtracted(source="pipeline")`` this row's
    original pipeline run never emitted — Tier 1 gated
    ``trigger_profile_clustering`` off before it could fire (its
    per-dispatch body-guard returns early when embedding is
    unavailable).

    ``event_id`` carries a ``backfill_`` prefix so ops can tell a
    synthesized run apart from a real one in logs / run records.
    """
    return EpisodeExtracted(
        event_id=f"backfill_{uuid4().hex}",
        memcell_id=raw["parent_id"],
        episode_entry_id=raw["entry_id"],
        episode_text=raw["episode"],
        episode_timestamp_ms=to_timestamp_ms(raw["timestamp"]),
        owner_id=raw["owner_id"],
        session_id=raw.get("session_id"),
        app_id=raw.get("app_id", "default"),
        project_id=raw.get("project_id", "default"),
        source="pipeline",
    )


def _agent_case_row_to_event(raw: dict[str, Any]) -> AgentCaseExtracted:
    """Synthesize the ``AgentCaseExtracted`` this row's original pipeline
    run never emitted, mirroring :func:`_episode_row_to_event`."""
    return AgentCaseExtracted(
        event_id=f"backfill_{uuid4().hex}",
        memcell_id=raw["parent_id"],
        case_entry_id=raw["entry_id"],
        task_intent=raw["task_intent"],
        quality_score=raw["quality_score"],
        case_timestamp_ms=to_timestamp_ms(raw["timestamp"]),
        agent_id=raw["owner_id"],
        app_id=raw.get("app_id", "default"),
        project_id=raw.get("project_id", "default"),
    )


def _report_emit_progress(presenter: BackfillPresenter, done: int, total: int) -> None:
    if done % _PROGRESS_EVERY == 0 or done == total:
        presenter.emit_progress(done, total)


async def _emit_synthetic_events(
    engine: OfflineEngine,
    episodes: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    presenter: BackfillPresenter,
) -> int:
    """Fan every row into ``engine`` as its own synthetic trigger event.

    Skip rows that a prior Phase-2 run already clustered — otherwise a
    rerun (after Ctrl-C, or ``everos cascade backfill --phase all --yes``
    on a partially-completed root) would re-cluster the same rows and
    grow cluster counts spuriously. ``cluster_repo.find_cluster_id_for_member``
    is O(log N) via the reverse index and is the exact primitive for
    this dedup. ``member_type`` values (``"episode"`` / ``"case"``) match
    what :func:`trigger_profile_clustering` and
    :func:`trigger_skill_clustering` insert on the write path — a mismatch
    here would silently disable the skip and re-open the double-cluster
    window.

    Episodes first, then agent cases — order doesn't affect correctness
    (each event routes to its own strategy independently); it only
    keeps the progress readout monotonic. The progress counter advances
    for skipped rows too so the readout matches the pre-scan estimate;
    ``_ClusterPhaseResult.events_emitted`` reflects "rows processed"
    (real emits + already-clustered skips), not "engine.emit calls".
    """
    total = len(episodes) + len(cases)
    emitted = 0
    for raw in episodes:
        # entry_id is only per-owner unique — scope the reverse lookup
        # so a same-day, same-seq episode under a different owner does
        # not falsely match this owner's cluster (or vice versa).
        existing = await cluster_repo.find_cluster_id_for_member(
            member_type="episode",
            member_id=raw["entry_id"],
            app_id=raw["app_id"],
            project_id=raw["project_id"],
            owner_id=raw["owner_id"],
        )
        if existing is None:
            await engine.emit(_episode_row_to_event(raw))
        emitted += 1
        _report_emit_progress(presenter, emitted, total)
    for raw in cases:
        existing = await cluster_repo.find_cluster_id_for_member(
            member_type="case",
            member_id=raw["entry_id"],
            app_id=raw["app_id"],
            project_id=raw["project_id"],
            owner_id=raw["owner_id"],
        )
        if existing is None:
            await engine.emit(_agent_case_row_to_event(raw))
        emitted += 1
        _report_emit_progress(presenter, emitted, total)
    return emitted


def ome_lock_is_free() -> bool:
    """Whether no other process holds the OME jobstore lock.

    ``False`` means a live ``everos server`` (or another exclusive CLI
    phase) is running against this memory root. Public entry point for
    commands that must not run concurrently with the daemon — notably
    ``cascade rebuild``, which drops and recreates the LanceDB tables
    under any cached handles a running daemon still holds.

    Same best-effort caveat as :func:`_probe_ome_lock_available`: the lock
    can be taken between this probe and the destructive step, so it is a
    guard against the common mistake, not a mutual-exclusion primitive.
    """
    return _probe_ome_lock_available()


def _probe_ome_lock_available() -> bool:
    """Probe whether the OME jobstore file lock is free.

    Backfill Phase 2/3 need exclusive access to
    ``<root>/.index/sqlite/ome.db.lock``. If another OfflineEngine
    (typically ``everos server start``) already holds it, the phase
    cannot proceed — better to detect that before printing the phase
    header and asking the user to confirm.

    Uses ``LOCK_EX | LOCK_NB`` briefly, then releases. Returns ``True``
    when acquire succeeded (lock was free at probe time) and ``False``
    when :class:`portalocker.LockException` fired (someone else holds it).

    Best-effort UX polish, not a correctness gate: the lock could become
    held between the probe and the actual :meth:`OfflineEngine.start`.
    That race path still surfaces :class:`EngineLockHeldError`, caught in
    the phase runners and reported through the same friendly path.
    """
    root = MemoryRoot.resolve()
    lock_path = Path(str(root.ome_db) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")  # noqa: SIM115
    try:
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException:
            return False
        portalocker.unlock(handle)
        return True
    finally:
        handle.close()


def _build_cluster_engine() -> OfflineEngine:
    """Construct (but do not start) the throw-away OME engine Phase 2 drives.

    ``memory`` may not import ``service`` (the layering rule forbids it —
    see ``.claude/rules/architecture.md``), so this cannot reuse
    ``service.memorize``'s process-wide engine singleton; it builds its
    own instance and registers only the two clustering strategies whose
    body-guards short-circuit under Tier 1 (no embedding provider).
    It shares the live engine's ``ome_db``
    jobstore path, so if a server is already running against the same
    memory root, ``engine.start()`` fails fast on the file lock instead
    of racing it — stop the server before running backfill.
    ``config_path`` is deliberately left unset: a one-shot migration has
    no use for hot-reloadable per-strategy overrides, and skipping it
    means this never depends on ``ome.toml`` existing.

    ``crash_recovery_enabled=False`` prevents this backfill-scoped engine
    from re-enqueueing the server's stale RUNNING rows into its own APS
    scheduler. The backfill engine only registers the Phase-2 clustering
    strategies, so a stale row for any other strategy (e.g.
    ``extract_atomic_facts``, ``extract_agent_case``) would hit
    ``StrategyRegistry.get`` with an unknown name and raise on dispatch,
    permanently marking the event CRASHED-but-not-recovered. Those rows
    stay put and are resumed on the next server restart, from an engine
    that actually knows those strategy names. See PR #361 review
    finding M10.
    """
    root = MemoryRoot.resolve()
    root.ome_db.parent.mkdir(parents=True, exist_ok=True)
    engine = OfflineEngine(
        config=OMEConfig(
            jobstore_path=root.ome_db,
            crash_recovery_enabled=False,
        )
    )
    engine.register(trigger_profile_clustering)
    engine.register(trigger_skill_clustering)
    return engine


async def _ensure_cluster_schema() -> None:
    """Create every sqlite table (incl. ``cluster``/``cluster_member``) if
    this is the first touch of this memory root's system db.

    Phase 1 only touches LanceDB, whose tables auto-create on
    ``get_table``. Phase 2 is the first backfill phase to write
    through ``cluster_repo``. The CLI ``backfill`` command wraps this
    coroutine in the cascade ``_runtime()`` context (same as the
    ``sync`` / ``status`` / ``fix`` siblings), which already runs
    ``metadata.create_all``; this call is defense-in-depth for other
    entry points that invoke the phase runners directly. Idempotent
    (``create_all`` no-ops on existing tables).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def _run_phase_clusters(
    *, auto_yes: bool, presenter: BackfillPresenter
) -> _ClusterPhaseResult:
    """Rebuild clusters for every episode / agent case via synthetic events.

    Scan every episode + agent case row → surface the estimate →
    confirm once → synthesize the ``EpisodeExtracted`` /
    ``AgentCaseExtracted`` event each row's original pipeline run
    would have emitted had the clustering strategies not been gated
    off under Tier 1 (their embed-requiring body-guards short-circuited
    the dispatch) → replay them through a dedicated
    engine that registers only those two (now-eligible, since embed
    is available) strategies → wait for the engine to drain.

    Idempotent: :func:`_emit_synthetic_events` filters each row through
    :meth:`cluster_repo.find_cluster_id_for_member` before emitting, so
    rerunning the phase (or triggering it from ``--phase all`` after a
    Ctrl-C interruption) skips rows already attached to a cluster.
    Cluster counts stop growing spuriously across reruns.

    Episode rows carrying ``parent_type == "cluster"`` are Reflection's
    merged episodes (``orchestrator._write_merged_episode``), not source
    pipeline events, and are excluded — mirrors the same
    ``parent_type == "memcell"`` filter idiom used by
    ``extract_user_profile._select_via_cluster``. Synthesizing an event
    for one would carry a bogus ``memcell_id`` (actually a cluster id)
    and defeat ``trigger_profile_clustering``'s own
    ``applies_to=lambda e: e.source == "pipeline"`` exclusion of
    Reflection output.
    """
    # Preflight capability + OME lock BEFORE any collection work. Both
    # are cheap checks; either failing makes the run impossible, so
    # doing them before ``_scan_all_rows`` (which is bounded but not
    # free) matches the Phase-1 defense-in-depth invariant. Capability
    # first: it's a permanent config gap, while the OME lock is
    # transient (stop the server and try again).
    capability = get_embedding_capability()
    if not capability.available:
        err = ProviderNotConfiguredError(provider="embedding", feature="backfill")
        presenter.capability_missing(
            provider="embedding", feature="backfill", message=str(err)
        )
        return _ClusterPhaseResult(blocked_by_capability="embedding")

    if not await anyio.to_thread.run_sync(_probe_ome_lock_available):
        presenter.server_running()
        return _ClusterPhaseResult(aborted=True, blocked_by_server=True)

    episodes = [
        row
        for row in await _scan_all_rows(Episode)
        if row.get("parent_type") == "memcell"
    ]
    cases = await _scan_all_rows(AgentCase)
    total = len(episodes) + len(cases)
    if total == 0:
        presenter.nothing_to_backfill(
            "Nothing to backfill. No episodes or agent cases found."
        )
        return _ClusterPhaseResult()

    presenter.estimate_clusters(len(episodes), len(cases))
    if not await presenter.confirm(
        f"proceed with {total:,} memories "
        f"({len(episodes):,} episodes, {len(cases):,} agent cases)",
        auto_yes=auto_yes,
    ):
        return _ClusterPhaseResult(aborted=True)

    await _ensure_cluster_schema()
    clusters_before = await cluster_repo.count()

    engine = _build_cluster_engine()
    try:
        await engine.start()
    except EngineLockHeldError:
        # Race window: preflight succeeded, but a server started between
        # probe and engine.start(). Same friendly path — no traceback.
        logger.warning("cascade_backfill_clusters_lock_race")
        presenter.server_running()
        return _ClusterPhaseResult(aborted=True, blocked_by_server=True)
    try:
        emitted = await _emit_synthetic_events(
            engine, episodes, cases, presenter=presenter
        )
        drained = await engine.wait_idle(timeout=_CLUSTER_WAIT_TIMEOUT_SECONDS)
        if not drained:
            logger.warning(
                "cascade_backfill_clusters_wait_timeout",
                timeout=_CLUSTER_WAIT_TIMEOUT_SECONDS,
            )
    finally:
        await engine.stop()

    clusters_after = await cluster_repo.count()
    presenter.phase_2_complete(clusters_before, clusters_after, emitted)
    return _ClusterPhaseResult(
        events_emitted=emitted,
        clusters_before=clusters_before,
        clusters_after=clusters_after,
        drained=drained,
    )


# ── Phase 3 (skills) — synthetic-event replay + cascade sync ────────────────

_SKILLS_WAIT_TIMEOUT_SECONDS = 300.0
"""Same rationale as :data:`_CLUSTER_WAIT_TIMEOUT_SECONDS`: cap on waiting
for the ephemeral engine to drain."""


@dataclasses.dataclass(frozen=True)
class _SkillSourceRow:
    """One (case, cluster) pairing Phase 3 replays as a synthetic
    ``SkillClusterUpdated`` event.

    Mirrors one row of the agent-case-kind cluster's membership
    (``ClusterMember`` via :meth:`cluster_repo.list_for_owner`) — the
    same shape :func:`trigger_skill_clustering` itself emits per fresh
    case, just synthesized after the fact for a case Phase 3 finds
    already clustered but not yet skill-extracted.
    """

    case_entry_id: str
    cluster_id: str
    agent_id: str
    app_id: str
    project_id: str


@dataclasses.dataclass
class _SkillPhaseResult:
    """Phase 3 outcome.

    ``events_emitted`` counts every synthesized ``SkillClusterUpdated``
    event fanned into the ephemeral engine — one per
    :class:`_SkillSourceRow`. ``skills_before`` / ``skills_after`` are
    the total ``agent_skill`` row count (:meth:`agent_skill_repo.count`)
    taken immediately before dispatch and after both the engine drains
    and the follow-up cascade sync runs (see :func:`_sync_new_skill_files`),
    so the caller reports real, already-searchable growth rather than a
    number that only becomes true once some future cascade run catches
    up. ``drained`` / ``aborted`` mirror :class:`_ClusterPhaseResult`.
    """

    events_emitted: int = 0
    skills_before: int = 0
    skills_after: int = 0
    drained: bool = True
    aborted: bool = False
    blocked_by_server: bool = False
    blocked_by_capability: str | None = None


async def _skill_md_exists_for_cluster(
    *,
    cluster_id: str,
    agent_id: str,
    app_id: str,
    project_id: str,
    memory_root: MemoryRoot,
) -> bool:
    """Check whether any ``SKILL.md`` under this agent's skills dir
    already carries ``cluster_id`` in its frontmatter.

    ``extract_agent_skill`` writes ``SKILL.md`` first and only reaches
    LanceDB via the cascade sync afterwards
    (see :func:`_sync_new_skill_files`). If a Phase-3 run is
    interrupted (Ctrl-C during ``engine.wait_idle`` or between
    ``engine.stop`` and the sync), some clusters will have their
    ``SKILL.md`` on disk but no LanceDB row. Round-1's idempotency
    probe checked only ``agent_skill_repo.count_in_cluster``, so a
    later Phase-3 would re-fire ``SkillClusterUpdated`` and produce a
    duplicate skill on the next drain. The disk check closes that
    window.

    ``skill_name`` is LLM-generated (see
    :class:`AgentSkillFrontmatter.name`) so the cluster→md-path
    mapping isn't stable — we scan the agent's ``skills/`` directory
    for any ``skill_*/SKILL.md`` whose frontmatter names this
    ``cluster_id``. This is O(existing skills for the agent), which
    is bounded by the agent's own history; Phase 3 already runs a
    per-cluster preflight, so an extra frontmatter parse per cluster
    is proportionate.
    """
    skills_dir = (
        memory_root.agents_dir(app_id, project_id)
        / agent_id
        / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
    )
    apath = anyio.Path(skills_dir)
    if not await apath.is_dir():
        return False
    async for skill_dir in apath.iterdir():
        if not await skill_dir.is_dir():
            continue
        if not skill_dir.name.startswith(AgentSkillFrontmatter.SKILL_DIR_PREFIX):
            continue
        md_path = skill_dir / AgentSkillFrontmatter.SKILL_MAIN_FILENAME
        if not await md_path.is_file():
            continue
        try:
            parsed = await MarkdownReader.read(Path(str(md_path)))
        except Exception as exc:
            # A malformed SKILL.md is not fatal — leave the cluster
            # ineligible for the disk-based skip so the scan falls
            # back to the LanceDB probe.
            logger.warning(
                "cascade_backfill_skill_md_parse_failed",
                md_path=str(md_path),
                error=repr(exc),
            )
            continue
        if parsed.frontmatter.get("cluster_id") == cluster_id:
            return True
    return False


async def _scan_skill_source() -> list[_SkillSourceRow]:
    """Enumerate every clustered agent case Phase 3 should (re-)replay.

    ``trigger_skill_clustering`` runs on the ``agent_case`` track
    (``owner_type == "agent"``, ``kind == "agent_case"`` — see
    ``Cluster.kind`` docstring); Phase 2's throwaway engine
    (:func:`_build_cluster_engine`) registers that strategy but not
    ``extract_agent_skill``, so every ``SkillClusterUpdated`` it emitted
    during Phase 2 had no listener and was dropped. This scan walks
    every agent-owned cluster :meth:`cluster_repo.list_distinct_owners`
    has ever seen and fans each cluster's members out into one row —
    Phase 3's job is to replay the event those members never got
    handled for.

    Idempotent: a cluster is skipped when EITHER
    ``agent_skill_repo.count_in_cluster`` > 0 (LanceDB knows about it)
    OR :func:`_skill_md_exists_for_cluster` returns ``True`` (a
    ``SKILL.md`` on disk names this cluster). The disk check closes
    the failure window where ``extract_agent_skill`` wrote the md but
    Phase 3 was interrupted before ``_sync_new_skill_files`` ran — a
    later re-run would otherwise re-extract every such cluster and
    double the skill count.
    """
    memory_root = MemoryRoot.resolve()
    rows: list[_SkillSourceRow] = []
    owners = await cluster_repo.list_distinct_owners()
    for owner_id, owner_type, app_id, project_id in owners:
        if owner_type != "agent":
            continue
        clusters = await cluster_repo.list_for_owner(
            owner_id, "agent_case", app_id=app_id, project_id=project_id
        )
        for cluster in clusters:
            assert cluster.id is not None  # persisted clusters always carry an id
            already_extracted = await agent_skill_repo.count_in_cluster(
                owner_id=owner_id, cluster_id=cluster.id
            )
            if already_extracted:
                continue
            if await _skill_md_exists_for_cluster(
                cluster_id=cluster.id,
                agent_id=owner_id,
                app_id=app_id,
                project_id=project_id,
                memory_root=memory_root,
            ):
                logger.info(
                    "cascade_backfill_skill_md_present_skip_recluster",
                    cluster_id=cluster.id,
                    agent_id=owner_id,
                )
                continue
            rows.extend(
                _SkillSourceRow(
                    case_entry_id=case_entry_id,
                    cluster_id=cluster.id,
                    agent_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                )
                for case_entry_id in cluster.members
            )
    return rows


def _skill_source_to_event(row: _SkillSourceRow) -> SkillClusterUpdated:
    """Synthesize the ``SkillClusterUpdated`` Phase 2's clustering pass
    emitted into a void — see :func:`_scan_skill_source`."""
    return SkillClusterUpdated(
        event_id=f"backfill_{uuid4().hex}",
        case_entry_id=row.case_entry_id,
        cluster_id=row.cluster_id,
        agent_id=row.agent_id,
        app_id=row.app_id,
        project_id=row.project_id,
    )


def _build_skill_engine() -> OfflineEngine:
    """Construct (but do not start) the throw-away OME engine Phase 3 drives.

    Mirrors :func:`_build_cluster_engine`, registering only
    ``extract_agent_skill`` — the strategy whose body-guard short-circuits
    under Tier 1 (no embedding provider) and the one Phase 2's own
    throwaway engine never registered.

    ``crash_recovery_enabled=False`` for the same reason as
    :func:`_build_cluster_engine`: this Phase-3 engine registers only
    ``extract_agent_skill``, so re-enqueueing a server's stale RUNNING
    row for any other strategy would raise on dispatch and permanently
    lose the event. See PR #361 review finding M10.
    """
    root = MemoryRoot.resolve()
    root.ome_db.parent.mkdir(parents=True, exist_ok=True)
    engine = OfflineEngine(
        config=OMEConfig(
            jobstore_path=root.ome_db,
            crash_recovery_enabled=False,
        )
    )
    engine.register(extract_agent_skill)
    return engine


async def _sync_new_skill_files() -> None:
    """Drain the cascade queue once so the ``SKILL.md`` files
    ``extract_agent_skill`` just wrote land in LanceDB.

    Phases 1 and 2 write their own storage directly (LanceDB ``vector``
    column; the sqlite ``cluster`` table) and are immediately consistent
    once their own call returns. Phase 3's strategy only writes markdown
    (:class:`AgentSkillWriter`) and relies on cascade for the LanceDB
    side — so, unlike the other two phases, it has to explicitly ask
    cascade to catch up rather than being done as soon as the engine
    drains, or ``skills_after`` would under-report until some later,
    unrelated cascade run happens to pick the files up.
    """
    # Scope the sync to ``agent_skill`` only: an unscoped sweep would
    # walk every registered kind including knowledge_document /
    # knowledge_topic, and if the process has embedding but not rerank,
    # the knowledge handlers are gated off — cascade would then mark
    # every unseen knowledge md as permanently failed. A backfill run
    # must not pollute unrelated kinds' queue state.
    orchestrator = CascadeOrchestrator(
        memory_root=MemoryRoot.resolve(), tokenizer=build_tokenizer()
    )
    await orchestrator.sync_once(kinds={"agent_skill"})


async def _run_phase_skills(
    *, auto_yes: bool, presenter: BackfillPresenter
) -> _SkillPhaseResult:
    """Extract agent skills from Phase 2's clustered agent cases.

    Preflight → sync orphan ``SKILL.md`` files → scan every agent-case
    cluster missing a skill → surface the estimate → confirm once →
    replay the ``SkillClusterUpdated`` event each clustered case never
    got handled for (see :func:`_scan_skill_source`) through a
    dedicated engine that registers only ``extract_agent_skill`` →
    wait for it to drain.

    Sync runs BEFORE :func:`_scan_skill_source` — not after the drain —
    for the mid-Phase-3 Ctrl-C recovery path: ``extract_agent_skill``
    writes ``SKILL.md`` first and only reaches LanceDB via cascade
    afterwards, so an interrupted Phase 3 can leave orphan ``SKILL.md``
    files on disk. Group F's on-disk idempotency probe
    (:func:`_skill_md_exists_for_cluster`) skips clusters whose
    ``SKILL.md`` already exists, so on a rerun ``_scan_skill_source``
    can legitimately return an empty list even though those md files
    aren't yet indexed. Running sync unconditionally, before the scan,
    guarantees those orphans get picked up regardless of whether the
    scan finds fresh work. :func:`_sync_new_skill_files` is a no-op
    when the cascade queue is empty, so the cost is one
    ``scan_once`` + ``drain_until_empty`` round-trip.
    """
    await _ensure_cluster_schema()

    # Preflight capability + OME lock BEFORE the (potentially large)
    # cluster + membership scan. Phase 3 depends on Phase 2's
    # clusters, which only exist if embed was available at OME
    # startup — the feature label reflects what Phase 3 actually does
    # (skill extraction) so the toml hint names the right context.
    # Ordering mirrors :func:`_run_phase_clusters`.
    capability = get_embedding_capability()
    if not capability.available:
        err = ProviderNotConfiguredError(
            provider="embedding", feature="skill_extraction_backfill"
        )
        presenter.capability_missing(
            provider="embedding",
            feature="skill_extraction_backfill",
            message=str(err),
        )
        return _SkillPhaseResult(blocked_by_capability="embedding")

    if not await anyio.to_thread.run_sync(_probe_ome_lock_available):
        presenter.server_running()
        return _SkillPhaseResult(aborted=True, blocked_by_server=True)

    # Recover orphan SKILL.md files a prior interrupted Phase 3 may
    # have left on disk (see docstring). Runs before the scan so the
    # on-disk idempotency probe can't strand those md files unindexed.
    await _sync_new_skill_files()

    source_rows = await _scan_skill_source()
    if not source_rows:
        presenter.nothing_to_backfill(
            "Nothing to backfill. No agent skill clusters found."
        )
        return _SkillPhaseResult()

    cluster_count = len({row.cluster_id for row in source_rows})
    presenter.estimate_skills(len(source_rows), cluster_count)
    if not await presenter.confirm(
        f"proceed with {len(source_rows):,} agent case(s) across "
        f"{cluster_count:,} cluster(s)",
        auto_yes=auto_yes,
    ):
        return _SkillPhaseResult(aborted=True)

    skills_before = await agent_skill_repo.count()

    engine = _build_skill_engine()
    try:
        await engine.start()
    except EngineLockHeldError:
        # Race window mirror of the Phase 2 handling in _run_phase_clusters.
        logger.warning("cascade_backfill_skills_lock_race")
        presenter.server_running()
        return _SkillPhaseResult(aborted=True, blocked_by_server=True)
    try:
        emitted = 0
        for row in source_rows:
            await engine.emit(_skill_source_to_event(row))
            emitted += 1
            _report_emit_progress(presenter, emitted, len(source_rows))
        drained = await engine.wait_idle(timeout=_SKILLS_WAIT_TIMEOUT_SECONDS)
        if not drained:
            logger.warning(
                "cascade_backfill_skills_wait_timeout",
                timeout=_SKILLS_WAIT_TIMEOUT_SECONDS,
            )
    finally:
        await engine.stop()

    # Sync of orphan SKILL.md files ran before the scan (see docstring);
    # the new files this drain just wrote also need to land in LanceDB
    # before ``skills_after`` is read, or the reported count would
    # under-report until some later, unrelated cascade run happens.
    await _sync_new_skill_files()
    skills_after = await agent_skill_repo.count()

    presenter.phase_3_complete(skills_before, skills_after, emitted)
    return _SkillPhaseResult(
        events_emitted=emitted,
        skills_before=skills_before,
        skills_after=skills_after,
        drained=drained,
    )


@dataclasses.dataclass
class _BackfillSummary:
    """Accumulates each phase's result as the orchestrator runs them.

    A field stays ``None`` for a phase that never ran (single-``--phase``
    invocation, or an abort/interrupt before that phase started) — the
    printed summary only shows a line for phases that actually executed,
    rather than padding unrun phases with fake zeros.
    """

    vectors: _PhaseResult | None = None
    clusters: _ClusterPhaseResult | None = None
    skills: _SkillPhaseResult | None = None


def _count_failed_rows(summary: _BackfillSummary) -> int:
    """Sum ``rows_failed`` across every phase that ran.

    Only Phase 1 exposes a row-level failure counter today — Phase 2
    (cluster formation) and Phase 3 (skill extraction) surface failures
    through OME strategy retries and logging, not as row-level counters
    on their result dataclasses. So only :class:`_PhaseResult` contributes.
    Keeping this helper standalone leaves an obvious extension point once
    Phase 2/3 grow their own ``rows_failed`` field.
    """
    return summary.vectors.rows_failed if summary.vectors else 0
