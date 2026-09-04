"""KindRecaller protocol + LanceDB row → Candidate helpers.

Every recaller exposes two callsites:

* :meth:`sparse_recall` — BM25 over the schema's ``*_tokens`` FTS column(s);
* :meth:`dense_recall`  — cosine ANN over the 1024-d ``vector`` column.

Both are filtered by the precompiled LanceDB ``where`` string and capped
at ``limit`` (the candidate pool size). The recaller does **not** apply
``radius``; that runs in the manager so the same value applies before
fusion / rerank.

A shared :class:`RecallerDeps` bundles the providers a recaller needs
at construction time (tokenizer for BM25 query, embedder is consumed
upstream by the manager so we keep deps minimal). The bundle keeps the
constructor signatures identical across the four LanceDB-backed
recallers so the orchestrator wiring stays uniform.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from everalgo.types import Candidate

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.index import Predicate

# Columns that should never travel through the ranker / shaper. ``vector``
# is huge (1024 floats); ``_distance`` belongs to LanceDB's query engine
# and is converted into ``score`` before the row leaves the recaller.
_NOISE_COLUMNS: frozenset[str] = frozenset(
    {"vector", "subject_vector", "_distance", "_score", "created_at", "updated_at"}
)


@dataclasses.dataclass(frozen=True)
class RecallerDeps:
    """Shared dependencies for every LanceDB-backed recaller.

    Frozen so the orchestrator can build one instance and hand it to
    every recaller without worrying about state divergence.
    """

    tokenizer: Tokenizer


@runtime_checkable
class KindRecaller(Protocol):
    """One business kind, BM25 + vector recall over its LanceDB table."""

    kind: ClassVar[str]
    """``episode`` / ``atomic_fact`` / ``agent_case`` / ``agent_skill``."""

    everalgo_memory_type: ClassVar[str]
    """``episodic`` / ``case`` / ``skill`` — passed to ``RankInput.memory_type``."""

    text_field: ClassVar[str]
    """Source column for cross-encoder rerank passages (display text)."""

    async def sparse_recall(
        self, query: str, where: Predicate, *, limit: int
    ) -> list[Candidate]: ...

    async def dense_recall(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]: ...


def row_to_candidate(
    row: dict[str, Any],
    *,
    source: str,
    score: float,
) -> Candidate:
    """Pack a LanceDB row dict into an everalgo ``Candidate``.

    The full row (minus noise columns) rides in ``metadata`` so the
    shaper can build the response DTO without going back to LanceDB.
    """
    rid = row.get("id")
    if not isinstance(rid, str):
        raise ValueError(f"row missing string 'id': {row!r}")
    metadata = {k: v for k, v in row.items() if k not in _NOISE_COLUMNS and k != "id"}
    return Candidate(
        id=rid,
        score=float(score),
        source=source,  # type: ignore[arg-type]  # "keyword" | "vector"
        metadata=metadata,
    )


def cosine_score_from_distance(distance: float | None) -> float:
    """Convert LanceDB cosine ``_distance`` → similarity in ``[0, 1]``.

    With ``metric='cosine'``, the engine emits ``distance = 1 - cos``,
    so similarity is its complement. ``None`` is treated as 0.0 (no
    score; lets BM25-only rows survive a merge).
    """
    if distance is None:
        return 0.0
    sim = 1.0 - float(distance)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim
