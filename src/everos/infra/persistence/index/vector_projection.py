"""Shared embedding sources and vector columns for indexing and repair.

Each derived-index table embeds one text into its ``vector`` column. Episodes
embed their body only: ``Episode.subject_vector`` stays in the schema, so
stores written before it was retired still open, but nothing fills or reads
it.
"""

from __future__ import annotations

__all__ = ["VECTOR_PROJECTIONS", "VectorProjection"]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from everos.memory.ports import Filter, is_null


@dataclass(frozen=True)
class VectorProjection:
    """Pure projection of index content onto its embedding column.

    Args:
        table: Index kind.
        text_of: Extract the embedding input from normalized content.
    """

    table: str
    text_of: Callable[[Mapping[str, Any]], str]
    column: str = "vector"

    def texts(self, content: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the embedding inputs in vector-column order."""
        return (self.text_of(content),)

    def null_filter(self) -> Filter:
        """Select rows still missing their vector."""
        return is_null(self.column)

    def values(
        self, vectors: Sequence[list[float] | None]
    ) -> dict[str, list[float] | None]:
        """Map ordered results onto the column, retaining an unavailable value."""
        return {self.column: vectors[0] if vectors else None}


def _skill_text(row: Mapping[str, Any]) -> str:
    return "\n".join(s for s in (row.get("name"), row.get("description")) if s)


VECTOR_PROJECTIONS: dict[str, VectorProjection] = {
    spec.table: spec
    for spec in (
        VectorProjection("episode", lambda r: r["episode"]),
        VectorProjection("atomic_fact", lambda r: r["fact"]),
        VectorProjection("agent_case", lambda r: r["task_intent"]),
        VectorProjection("agent_skill", _skill_text),
        VectorProjection("knowledge_topic", lambda r: r["summary"]),
    )
}
