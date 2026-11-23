"""Tenant boundaries for safely merged execution graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smartcomment.runtime.variable import RuntimeVariable


def same_tenant(node: RuntimeVariable, query: RuntimeVariable) -> bool:
    """Unmerged graphs share scope; merged records declare it explicitly."""
    scope = query.metadata.get("tenant_scope")
    return scope is None or node.metadata.get("tenant_scope") == scope
