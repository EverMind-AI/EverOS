"""LanceDB repositories, one per business table.

Each is built over a table source -- the application's
:class:`~everos.infra.persistence.lancedb.LanceDbConnection` -- by
:class:`~everos.infra.persistence.lancedb.LanceIndexBackend`, and opens its
table on first use.
"""

from ._contract import LanceIndexContract as LanceIndexContract
from ._contract import build_or_query as build_or_query
from ._contract import render_predicate as render_predicate

__all__ = [
    "LanceIndexContract",
    "build_or_query",
    "render_predicate",
]
