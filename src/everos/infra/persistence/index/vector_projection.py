"""Shared embedding sources and vector columns for indexing and repair."""

from __future__ import annotations

__all__ = ["VECTOR_PROJECTIONS", "VectorProjection"]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from everos.memory.ports import Filter, any_of, is_null


@dataclass(frozen=True)
class VectorProjection:
    """Pure projection of index content onto its embedding columns.

    Args:
        table: Index kind.
        text_of: Extract the primary embedding input from normalized content.
        subject_of: Extract an optional independent embedding input.
    """

    table: str
    text_of: Callable[[Mapping[str, Any]], str]
    subject_of: Callable[[Mapping[str, Any]], str | None] | None = None
    primary_column: str = "vector"
    subject_column: str = "subject_vector"

    def texts(self, content: Mapping[str, Any]) -> tuple[str, ...]:
        """Return inputs in vector-column order, omitting an absent subject."""
        primary = self.text_of(content)
        subject = self.subject_of(content) if self.subject_of else None
        return (primary, subject) if subject is not None else (primary,)

    def null_filter(self) -> Filter:
        """Select potentially incomplete rows, including absent subjects."""
        primary = is_null(self.primary_column)
        if self.subject_of is not None:
            return any_of(primary, is_null(self.subject_column))
        return primary

    def missing_sides(
        self, row: Mapping[str, Any], subject: str | None
    ) -> tuple[bool, bool]:
        """Distinguish a failed subject embedding from a legitimately absent one."""
        return (
            row.get(self.primary_column) is None,
            self.subject_of is not None
            and subject is not None
            and row.get(self.subject_column) is None,
        )

    def values(
        self, vectors: Sequence[list[float] | None]
    ) -> dict[str, list[float] | None]:
        """Map ordered results onto columns, retaining unavailable values."""
        values = {self.primary_column: vectors[0] if vectors else None}
        if self.subject_of is not None:
            values[self.subject_column] = vectors[1] if len(vectors) > 1 else None
        return values


def _skill_text(row: Mapping[str, Any]) -> str:
    return "\n".join(s for s in (row.get("name"), row.get("description")) if s)


VECTOR_PROJECTIONS: dict[str, VectorProjection] = {
    spec.table: spec
    for spec in (
        VectorProjection(
            "episode", lambda r: r["episode"], lambda r: r.get("subject") or None
        ),
        VectorProjection("atomic_fact", lambda r: r["fact"]),
        VectorProjection("foresight", lambda r: r["foresight"]),
        VectorProjection("agent_case", lambda r: r["task_intent"]),
        VectorProjection("agent_skill", _skill_text),
        VectorProjection("knowledge_topic", lambda r: r["summary"]),
    )
}
