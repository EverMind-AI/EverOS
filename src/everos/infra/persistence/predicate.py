"""Backend-neutral predicate tree for rebuildable derived indexes.

Application and memory code construct these nodes.  A storage adapter owns
the rendering into its physical query language, so backend syntax never leaks
above :mod:`everos.infra.persistence`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

type Scalar = str | int | float | bool | dt.datetime
type ComparisonOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte"]


class Predicate:
    """Marker base class for derived-index predicates."""


@dataclass(frozen=True)
class Comparison(Predicate):
    field: str
    operator: ComparisonOperator
    value: Scalar


@dataclass(frozen=True)
class In(Predicate):
    field: str
    values: tuple[Scalar, ...]


@dataclass(frozen=True)
class Contains(Predicate):
    field: str
    value: str


@dataclass(frozen=True)
class IsNull(Predicate):
    field: str


@dataclass(frozen=True)
class All(Predicate):
    children: tuple[Predicate, ...]


@dataclass(frozen=True)
class AnyOf(Predicate):
    children: tuple[Predicate, ...]


def compare(field: str, operator: ComparisonOperator, value: Scalar) -> Predicate:
    return Comparison(field, operator, value)


def eq(field: str, value: Scalar) -> Predicate:
    return compare(field, "eq", value)


def ne(field: str, value: Scalar) -> Predicate:
    return compare(field, "ne", value)


def gt(field: str, value: Scalar) -> Predicate:
    return compare(field, "gt", value)


def gte(field: str, value: Scalar) -> Predicate:
    return compare(field, "gte", value)


def lt(field: str, value: Scalar) -> Predicate:
    return compare(field, "lt", value)


def lte(field: str, value: Scalar) -> Predicate:
    return compare(field, "lte", value)


def one_of(field: str, values: list[Scalar] | tuple[Scalar, ...]) -> Predicate:
    if not values:
        raise ValueError("one_of requires at least one value")
    return In(field, tuple(values))


def contains(field: str, value: str) -> Predicate:
    return Contains(field, value)


def is_null(field: str) -> Predicate:
    return IsNull(field)


def all_of(*predicates: Predicate | None) -> Predicate:
    children: list[Predicate] = []
    for predicate in predicates:
        if predicate is None:
            continue
        if isinstance(predicate, All):
            children.extend(predicate.children)
        else:
            children.append(predicate)
    return All(tuple(children))


def any_of(*predicates: Predicate | None) -> Predicate:
    children: list[Predicate] = []
    for predicate in predicates:
        if predicate is None:
            continue
        if isinstance(predicate, AnyOf):
            children.extend(predicate.children)
        else:
            children.append(predicate)
    return AnyOf(tuple(children))


__all__ = [
    "All",
    "AnyOf",
    "Comparison",
    "Contains",
    "In",
    "IsNull",
    "Predicate",
    "Scalar",
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
]
