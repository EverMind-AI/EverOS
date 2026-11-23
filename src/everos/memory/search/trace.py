"""Request-level search-stage trace collector (MemTrace support).

The MemTrace tracer (:mod:`everos.infra.memtrace`) records the candidate
pool at each stage of the search pipeline so the exported execution graph
can answer "was the evidence in the recall pool, did it survive fusion,
was it returned". To keep the memory layer free of any tracing
dependency, this module exposes a zero-dependency collection protocol:

* :func:`record_stage` is called by the search manager at each pipeline
  stage. Without an installed collector it is a no-op; it never raises,
  so tracing can never break the search path.
* :func:`collect_search_trace` is entered once per ``/memory/search``
  request by the service layer and returns the ordered stage records.

Isolation is by :class:`contextvars.ContextVar`: concurrent search
requests (each entering its own ``collect_search_trace``) never share a
collector, while the ``asyncio.gather`` fan-out inside one request
inherits it, so episode / agent-case / agent-skill stages of the same
request land in one list.

Stage ``phase`` vocabulary: ``sparse`` / ``dense`` (per-route recall),
``single_route`` (KEYWORD / VECTOR one-route recall), ``maxsim_atomic``
(VECTOR episode max-pool; also the atomic-fact lane of episode HYBRID),
``rrf`` / ``hierarchy`` / ``filtered``
(episode HYBRID hierarchy pipeline), ``fusion`` (everalgo rank output),
``bridged_dense`` (case→skill bridge merged dense pool), ``agentic``.
``kind`` is the recaller business kind: ``episode`` / ``agent_case`` /
``agent_skill``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from everos.core.utils.datetime import get_utc_now

# Per-stage candidate cap — mirrors the pool-size ceiling used by the
# EverOS-1 inline instrumentation; keeps the exported graph bounded when
# unlimited mode recalls pools of 100+ candidates.
STAGE_CANDIDATE_CAP = 50


@dataclass(frozen=True)
class CandidateBrief:
    """One candidate reduced to its identity triple (id, score, source)."""

    id: str
    score: float
    source: str


@dataclass(frozen=True)
class SearchStageRecord:
    """The candidate pool observed at one stage of the search pipeline.

    ``total_count`` is the pool size before the ``STAGE_CANDIDATE_CAP``
    truncation (``-1`` when the source iterable exposed no length), so
    consumers can tell a capped pool snapshot from a complete one.
    """

    phase: str
    kind: str
    candidates: tuple[CandidateBrief, ...]
    total_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stage_id: str = ""
    input_stage_ids: tuple[str, ...] | None = None


_STAGE_COLLECTOR: ContextVar[list[SearchStageRecord] | None] = ContextVar(
    "everos_search_stage_collector", default=None
)


def record_stage(
    phase: str,
    kind: str,
    candidates: Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    inputs: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Append one stage record to the active collector, if any.

    No-op when no collector is installed (the default outside a
    ``collect_search_trace`` scope); never raises — a malformed candidate
    or collector failure must not break search.

    Args:
        phase (`str`):
            Pipeline phase name (see module docstring for the vocabulary).
        kind (`str`):
            Business kind of the recalled candidates.
        candidates (`Sequence[Any]`):
            Candidates at this stage; each is reduced to its
            ``id`` / ``score`` / ``source`` triple (attributes read
            defensively) and capped at :data:`STAGE_CANDIDATE_CAP`.
        metadata (`Mapping[str, Any] | None`, optional):
            Stage annotations (e.g. ``fusion_mode``, ``enable_rerank``).
    """
    collector = _STAGE_COLLECTOR.get()
    if collector is None:
        return
    try:
        try:
            total_count = len(candidates)
        except TypeError:
            total_count = -1  # The iterable exposed no length.
        briefs = tuple(
            CandidateBrief(
                id=str(getattr(candidate, "id", "")),
                score=float(getattr(candidate, "score", 0.0) or 0.0),
                source=str(getattr(candidate, "source", "") or ""),
            )
            for candidate in candidates[:STAGE_CANDIDATE_CAP]
        )
        stage_id = f"{kind}:{phase}:{len(collector)}"
        resolved_inputs: list[str] = []
        if inputs is not None:
            for input_kind, input_phase in inputs:
                parent = next(
                    (
                        s
                        for s in reversed(collector)
                        if s.kind == input_kind and s.phase == input_phase
                    ),
                    None,
                )
                if parent is None:
                    raise ValueError("Missing input search stage.")
                resolved_inputs.append(parent.stage_id)
        collector.append(
            SearchStageRecord(
                phase=phase,
                kind=kind,
                candidates=briefs,
                total_count=total_count,
                metadata=dict(metadata) if metadata else {},
                stage_id=stage_id,
                input_stage_ids=tuple(resolved_inputs) if inputs is not None else None,
            )
        )
    except Exception:  # tracing must never break the search path
        return


def record_branch(
    function: str,
    kind: str,
    results: Sequence[Any],
    *,
    status: str = "success",
    **metadata: Any,
) -> None:
    """Record an independent KV/buffer branch, including skips and empty reads.

    DTO snapshots are evaluated only with an active request collector. These
    branches depend on the request, never on the episode ranking stages.
    """
    if _STAGE_COLLECTOR.get() is None:
        return
    try:
        record_stage(
            function,
            kind,
            results,
            inputs=[],
            metadata={
                "function": function,
                "status": status,
                "observed_at": get_utc_now().isoformat(),
                "items": [r.model_dump(mode="json") for r in results],
                **metadata,
            },
        )
    except Exception:
        return


@contextmanager
def collect_search_trace() -> Iterator[list[SearchStageRecord]]:
    """Install a fresh stage collector for one search request.

    Yields the list that :func:`record_stage` appends to; the collector
    is uninstalled on exit, so nested or subsequent requests always start
    from an empty record.

    Returns:
        `Iterator[list[SearchStageRecord]]`:
            The ordered stage records collected within the scope.
    """
    stages: list[SearchStageRecord] = []
    token = _STAGE_COLLECTOR.set(stages)
    try:
        yield stages
    finally:
        _STAGE_COLLECTOR.reset(token)
