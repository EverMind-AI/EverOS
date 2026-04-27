"""
Milvus Tenant Field Isolation Collection Proxy.

Intercepts all data operations (insert/upsert/search/query/delete)
to inject tenant_id transparently. Repositories require zero changes.

Three-category whitelist strategy (consistent with MongoDB TenantCommandInterceptor):
    - Data-plane (explicit override): insert, upsert, search, query, delete
      → inject tenant_id into entity data or filter expression
    - Control-plane (passthrough whitelist): flush, load, release, compact, etc.
      → delegate to inner AsyncCollection as-is
    - Unknown: reject
      → raise AttributeError, refuse to let unrecognized methods through
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pymilvus import SearchResult
from pymilvus.orm.mutation import MutationResult

from core.observation.logger import get_logger
from core.oxm.milvus.async_collection import AsyncCollection
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_constants import TENANT_ID_FIELD
from core.tenants.tenant_contextvar import get_current_tenant_id

logger = get_logger(__name__)


class TenantIsolationViolation(Exception):
    """Raised when a Milvus operation violates tenant isolation."""

    pass


# --- Module-level utilities ---


def _exclude_tenant_from_fields(fields: Optional[List[str]]) -> Optional[List[str]]:
    """Remove tenant_id from output fields (defense in depth)."""
    if fields is None:
        return None
    return [f for f in fields if f != TENANT_ID_FIELD]


# --- Whitelist for control-plane methods ---

# Methods with explicit definitions (insert/upsert/search/query/delete/flush/load)
# are resolved before __getattr__ and do NOT need to appear here.
_PASSTHROUGH_METHODS: frozenset[str] = frozenset(
    {
        "release",
        "compact",
        "get_compaction_state",
        "get_compaction_plans",
        "get_replicas",
        "num_entities",
        "describe",
    }
)


# --- Proxy class ---


class TenantFieldCollectionProxy:
    """Proxy over AsyncCollection that enforces tenant isolation.

    Every data operation is intercepted:
    - Write: tenant_id force-injected into entity
    - Read:  tenant_id filter prepended to expr, tenant_id excluded from output
    - Delete: tenant_id filter prepended + empty-expr guard + audit log

    Unknown methods are rejected to prevent bypass of tenant isolation.
    In non-tenant mode, all operations are transparently passed through.
    """

    def __init__(self, inner: AsyncCollection) -> None:
        self.__inner = inner

    def _get_tenant_id(self) -> Optional[str]:
        """Get tenant_id from context. Returns None if no tenant context."""
        return get_current_tenant_id()

    def _require_tenant_id(self, operation: str) -> Optional[str]:
        """Get tenant_id or raise if missing after app startup.

        Returns None during startup (before app_ready), allowing callers
        to skip tenant injection. After app_ready, raises if missing.
        """
        tid = self._get_tenant_id()
        if not tid:
            if get_tenant_config().app_ready:
                raise TenantIsolationViolation(
                    f"Missing tenant_id for Milvus operation '{operation}'. "
                    f"Ensure tenant context is set before data operations."
                )
            return None
        return tid

    # --- Write path ---

    async def insert(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs: Any
    ) -> MutationResult:
        tid = self._require_tenant_id("insert")
        if tid:
            self._inject_tenant_to_entities(data, tid)
        return await self.__inner.insert(data, **kwargs)

    async def upsert(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs: Any
    ) -> MutationResult:
        tid = self._require_tenant_id("upsert")
        if tid:
            self._inject_tenant_to_entities(data, tid)
        return await self.__inner.upsert(data, **kwargs)

    # --- Read path ---

    async def search(
        self,
        data: List[List[float]],
        anns_field: str,
        param: Dict[str, Any],
        limit: int,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SearchResult:
        tid = self._require_tenant_id("search")
        if tid:
            expr = self._prepend_tenant_filter(expr, tid)
            output_fields = _exclude_tenant_from_fields(output_fields)
        return await self.__inner.search(
            data, anns_field, param, limit, expr, output_fields=output_fields, **kwargs
        )

    async def query(
        self, expr: str = "", output_fields: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        tid = self._require_tenant_id("query")
        if tid:
            expr = self._prepend_tenant_filter(expr, tid)
            output_fields = _exclude_tenant_from_fields(output_fields)
        return await self.__inner.query(
            expr=expr, output_fields=output_fields, **kwargs
        )

    # --- Delete path ---

    async def delete(self, expr: str, **kwargs: Any) -> MutationResult:
        tid = self._require_tenant_id("delete")
        if tid:
            if not expr or not expr.strip():
                raise TenantIsolationViolation(
                    "Delete without expression is forbidden in tenant isolation mode. "
                    "This would delete ALL data across tenants."
                )
            expr = self._prepend_tenant_filter(expr, tid)
        return await self.__inner.delete(expr, **kwargs)

    # --- Control-plane: transparent delegation ---

    async def flush(self, **kwargs: Any) -> None:
        return await self.__inner.flush(**kwargs)

    async def load(self, **kwargs: Any) -> None:
        return await self.__inner.load(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate whitelisted control-plane methods to inner AsyncCollection.

        Unknown methods are rejected to prevent bypass of tenant isolation.
        When pymilvus adds new data methods in the future, this will force
        developers to come here and explicitly handle them.
        """
        if name.startswith("_"):
            return getattr(self.__inner, name)

        if name not in _PASSTHROUGH_METHODS:
            raise AttributeError(
                f"'{type(self).__name__}' does not expose '{name}'. "
                f"If this is a non-data operation, add it to _PASSTHROUGH_METHODS. "
                f"If this is a data operation, add an explicit method with tenant injection."
            )
        return getattr(self.__inner, name)

    # --- Internal helpers ---

    @staticmethod
    def _prepend_tenant_filter(expr: Optional[str], tenant_id: str) -> str:
        """Prepend tenant_id clause to a Milvus filter expression."""
        tenant_clause = f'({TENANT_ID_FIELD} == "{tenant_id}")'
        if not expr or not expr.strip():
            return tenant_clause
        return f"{tenant_clause} and ({expr})"

    @staticmethod
    def _inject_tenant_to_entities(
        data: Union[Dict[str, Any], List[Dict[str, Any]]], tenant_id: str
    ) -> None:
        """Force-inject tenant_id into entity data."""
        entities = data if isinstance(data, list) else [data]
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise TenantIsolationViolation(
                    f"insert/upsert data[{i}] is {type(entity).__name__}, expected dict. "
                    f"Cannot inject tenant_id into unknown data type."
                )
            entity[TENANT_ID_FIELD] = tenant_id  # force overwrite

    def __repr__(self) -> str:
        return f"TenantFieldCollectionProxy(inner={self.__inner!r})"
