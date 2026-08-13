"""Adapter: ``/get`` filter compilation reuses :mod:`memory.search`.

After the 2026-05-16 decision to lift the AND/OR restriction
(``opensource`` memsys's MongoDB-backed ``/get`` always supported
combinators — the wiki appendix C rule was an unmotivated narrowing),
``/get`` and ``/search`` share the same filter DSL **and** the same
compile path. This module is a thin re-export so ``memory.get``
keeps a stable public name (``compile_filters_for_get``) even if the
underlying compile primitive ever gets renamed.

See :func:`everos.memory.search.filters.compile_filters` for the full
semantics (``owner_id`` / ``owner_type`` injection, ``ALLOWED_FIELDS``,
operator allow-list, timestamp / array-column rendering).
"""

from __future__ import annotations

from everos.memory.search import FilterNode, compile_filters


def compile_filters_for_get(
    filters: FilterNode | None,
    *,
    owner_id: str,
    owner_type: str,
    app_id: str = "default",
    project_id: str = "default",
) -> str:
    """Compile ``/get`` filters via the shared ``compile_filters`` path.

    Kept as a named wrapper so ``memory.get`` consumers depend on a
    stable name rather than reaching into ``memory.search``.
    """
    return compile_filters(
        filters,
        owner_id=owner_id,
        owner_type=owner_type,
        app_id=app_id,
        project_id=project_id,
    )
