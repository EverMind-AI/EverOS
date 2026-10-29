"""Backend-neutral query vocabulary shared by every storage port.

``Filter`` is a small predicate tree the domain builds; each index adapter
renders it into its own physical query language, so backend syntax never
leaks into the domain. The value objects next to it name what the ports
pass around: the ownership ``Scope`` a record lives in, the ``SessionKey``
a conversation buffer lives under, the identity a write returns
(``MemoryRef``), a persisted domain object together with that identity
(``Stored``), and a raw search hit (``IndexHit``).
"""

from __future__ import annotations

__all__ = [
    "All",
    "AnyOf",
    "Comparison",
    "Contains",
    "Filter",
    "In",
    "IndexHit",
    "IsNull",
    "MemoryRef",
    "OwnerType",
    "Page",
    "PageResult",
    "Scalar",
    "Scope",
    "SessionKey",
    "Stored",
    "all_of",
    "any_of",
    "compare",
    "contains",
    "eq",
    "gt",
    "gte",
    "is_null",
    "lt",
    "lte",
    "ne",
    "one_of",
    "primary_key",
]

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

type Scalar = str | int | float | bool | dt.datetime
type ComparisonOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte"]
type OwnerType = Literal["user", "agent"]


# ── Filter tree ──────────────────────────────────────────────────────────


class Filter:
    """Marker base class for backend-neutral filter nodes.

    Field names are domain attribute names; an adapter maps them onto its
    physical columns.
    """


@dataclass(frozen=True)
class Comparison(Filter):
    field: str
    operator: ComparisonOperator
    value: Scalar


@dataclass(frozen=True)
class In(Filter):
    field: str
    values: tuple[Scalar, ...]


@dataclass(frozen=True)
class Contains(Filter):
    field: str
    value: str


@dataclass(frozen=True)
class IsNull(Filter):
    field: str


@dataclass(frozen=True)
class All(Filter):
    children: tuple[Filter, ...]

    def __post_init__(self) -> None:
        _reject_empty(self)


@dataclass(frozen=True)
class AnyOf(Filter):
    children: tuple[Filter, ...]

    def __post_init__(self) -> None:
        _reject_empty(self)


def _reject_empty(group: All | AnyOf) -> None:
    """Forbid an empty group at construction, not just in the factories.

    Adapters render an empty group as an empty filter, and the two backends
    then disagree: LanceDB raises a raw SQL parse error, Milvus drops the
    filter and matches every row. Either way the meaning is wrong — an empty
    ``AnyOf`` means "match nothing" — and on an index delete the Milvus
    reading is a silent table wipe. ``All`` and ``AnyOf`` are public, so
    guarding only :func:`all_of` / :func:`any_of` leaves the hole open.
    """
    if not group.children:
        raise ValueError(f"{type(group).__name__} requires at least one child")


def compare(field: str, operator: ComparisonOperator, value: Scalar) -> Filter:
    return Comparison(field, operator, value)


def eq(field: str, value: Scalar) -> Filter:
    return compare(field, "eq", value)


def ne(field: str, value: Scalar) -> Filter:
    return compare(field, "ne", value)


def gt(field: str, value: Scalar) -> Filter:
    return compare(field, "gt", value)


def gte(field: str, value: Scalar) -> Filter:
    return compare(field, "gte", value)


def lt(field: str, value: Scalar) -> Filter:
    return compare(field, "lt", value)


def lte(field: str, value: Scalar) -> Filter:
    return compare(field, "lte", value)


def one_of(field: str, values: list[Scalar] | tuple[Scalar, ...]) -> Filter:
    if not values:
        raise ValueError("one_of requires at least one value")
    return In(field, tuple(values))


def contains(field: str, value: str) -> Filter:
    return Contains(field, value)


def is_null(field: str) -> Filter:
    return IsNull(field)


def all_of(*filters: Filter | None) -> Filter:
    """AND the given filters, flattening nested ``All`` and dropping ``None``.

    An empty result is rejected rather than rendered: adapters emit ``""`` for
    an empty group, and an empty filter means *match every row* — a silent
    "delete everything" if it ever reached an index delete.
    """
    children: list[Filter] = []
    for node in filters:
        if node is None:
            continue
        if isinstance(node, All):
            children.extend(node.children)
        else:
            children.append(node)
    if not children:
        raise ValueError("all_of requires at least one non-None predicate")
    return All(tuple(children))


def any_of(*filters: Filter | None) -> Filter:
    """OR the given filters, flattening nested ``AnyOf`` and dropping ``None``.

    Empty is rejected for the same reason as :func:`all_of`, and the inversion
    is worse here: an empty OR means "match nothing" but would render as
    "match everything".
    """
    children: list[Filter] = []
    for node in filters:
        if node is None:
            continue
        if isinstance(node, AnyOf):
            children.extend(node.children)
        else:
            children.append(node)
    if not children:
        raise ValueError("any_of requires at least one non-None predicate")
    return AnyOf(tuple(children))


def primary_key(app_id: str, project_id: str, row_id: str) -> Filter:
    """One record's ``(app_id, project_id, id)`` key as a filter.

    ``id`` is unique only within an ``(app_id, project_id)`` space, so a
    record is addressed by all three — never by ``id`` alone.
    """
    return all_of(eq("app_id", app_id), eq("project_id", project_id), eq("id", row_id))


# ── Value objects ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scope:
    """The ownership space one memory record lives in."""

    app_id: str
    project_id: str
    owner_id: str
    owner_type: OwnerType

    def to_filter(self) -> Filter:
        """The four equality conditions that confine a query to this scope."""
        return all_of(
            eq("owner_id", self.owner_id),
            eq("owner_type", self.owner_type),
            eq("app_id", self.app_id),
            eq("project_id", self.project_id),
        )


@dataclass(frozen=True)
class SessionKey:
    """The conversation a message buffer or memcell belongs to.

    Session state has no owner: attribution happens after boundary
    detection, so these records are keyed by space and session only.
    """

    app_id: str
    project_id: str
    session_id: str


@dataclass(frozen=True)
class Page:
    """One page of a sorted listing; ``number`` is 1-based."""

    sort_by: str
    number: int = 1
    size: int = 20
    descending: bool = True


@dataclass(frozen=True)
class PageResult[T]:
    """A page of items plus the total number of matches."""

    items: list[T]
    total: int


@dataclass(frozen=True)
class MemoryRef:
    """Identity of a persisted record.

    Attributes:
        id: Global key of the record in its index (``<owner_id>_<entry_id>``
            for daily-log kinds).
        entry_id: Identity inside the owning markdown file.
        md_path: Markdown file holding the record, relative to the memory
            root.
    """

    id: str
    entry_id: str
    md_path: str


@dataclass(frozen=True)
class Stored[T]:
    """A domain object that has been persisted, with its identity and scope."""

    ref: MemoryRef
    scope: Scope
    item: T


@dataclass(frozen=True)
class IndexHit:
    """One raw hit from a derived-index search.

    Attributes:
        id: Global key of the matched record.
        score: Backend relevance score when the query produced one.
        fields: The record's columns by name, including backend scoring
            columns such as ``_score`` / ``_distance``.
    """

    id: str
    score: float | None
    fields: Mapping[str, Any]
