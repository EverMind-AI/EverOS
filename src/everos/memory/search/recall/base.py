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
from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery

try:
    from lancedb.query import Occur
except ImportError:  # pragma: no cover — fallback for older LanceDB layouts
    from lancedb._lancedb import Occur  # type: ignore[attr-defined,no-redef]

from everos.component.tokenizer import Tokenizer

# Columns that should never travel through the ranker / shaper. ``vector``
# is huge (1024 floats); ``_distance`` belongs to LanceDB's query engine
# and is converted into ``score`` before the row leaves the recaller.
_NOISE_COLUMNS: frozenset[str] = frozenset(
    {"vector", "_distance", "_score", "created_at", "updated_at"}
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
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]: ...

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
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


def tokenize_query(tokenizer: Tokenizer, query: str) -> str:
    """Run the configured tokenizer over the query and join with spaces.

    Cascade joins tokens with a single space when writing the
    ``*_tokens`` columns; LanceDB FTS expects a whitespace-tokenised
    query string. Same function on both sides keeps BM25 scoring
    symmetric.

    Note: prefer :func:`build_or_query` for new code — it sidesteps
    the tantivy implicit-AND query-parser pitfall where a single
    IDF≈0 token (e.g. an owner's own name on the owner's partition)
    poisons the whole query into zero hits.
    """
    tokens = tokenizer.tokenize(query)
    return " ".join(t for t in tokens if t)


def build_or_query(
    tokenizer: Tokenizer,
    query: str,
    column: str,
) -> FullTextQuery | None:
    """Tokenise ``query`` and wrap in an OR-mode FTS query.

    Mirrors the enterprise ES design
    (``bool.should + minimum_should_match=1``):

    - jieba tokenises the user query.
    - Each token becomes its own :class:`MatchQuery` clause.
    - Clauses combine under :class:`BooleanQuery` with
      :attr:`Occur.SHOULD` so any single matching token surfaces
      the document.
    - LanceDB still computes a proper joint BM25 score from all
      clauses; tokens with IDF ≈ 0 contribute ~0 but no longer
      poison the rest of the query.

    Returns ``None`` when the query tokenises to nothing (the
    caller must guard against this — there's no useful BM25 query
    over an empty token set).

    Single-token queries collapse to a bare :class:`MatchQuery`
    (skipping the boolean wrapper) so the FTS engine doesn't pay
    for an unnecessary boolean layer.
    """
    tokens = [t for t in tokenizer.tokenize(query) if t]
    if not tokens:
        return None
    if len(tokens) == 1:
        return MatchQuery(tokens[0], column=column)
    clauses: list[tuple[Occur, FullTextQuery]] = [
        (Occur.SHOULD, MatchQuery(t, column=column)) for t in tokens
    ]
    return BooleanQuery(clauses)


def build_or_query_multi_column(
    tokenizer: Tokenizer,
    query: str,
    columns: Sequence[str],
) -> dict[str, FullTextQuery] | None:
    """Same as :func:`build_or_query` but emit one FTS query per column.

    ``MatchQuery`` is bound to a single column, and LanceDB FTS only
    searches one column per ``nearest_to_text`` call. Dual-column
    kinds (``agent_case`` over ``task_intent_tokens`` /
    ``approach_tokens``, ``agent_skill`` over
    ``description_tokens`` / ``content_tokens``, etc.) need one
    OR-bundle per column and merge the results in the caller.

    Returns ``None`` on empty tokenisation; otherwise a dict
    ``{column: FullTextQuery}`` ready to feed into separate
    ``nearest_to_text`` calls.
    """
    tokens = [t for t in tokenizer.tokenize(query) if t]
    if not tokens:
        return None
    out: dict[str, FullTextQuery] = {}
    for col in columns:
        if len(tokens) == 1:
            out[col] = MatchQuery(tokens[0], column=col)
        else:
            out[col] = BooleanQuery(
                [(Occur.SHOULD, MatchQuery(t, column=col)) for t in tokens]
            )
    return out
