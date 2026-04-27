"""
Elasticsearch Tenant Isolation

Single-file implementation of tenant isolation for Elasticsearch, consisting of
three logical sections:

    Section 1 — Query Utilities
        Pure functions that inject tenant_id filters into various ES query structures.
        No state, no side effects.

    Section 2 — Interceptor (Layer 1)
        TenantAwareAsyncElasticsearch: subclasses AsyncElasticsearch, overrides
        perform_request() — the single chokepoint ALL ES API calls go through.
        Injects tenant_id filter/field/routing based on endpoint_id.

    Section 3 — Guard Transport (Layer 2)
        TenantGuardTransport: subclasses AsyncTransport, independent verification
        layer. Does NOT modify data — only checks body structure and raises alarm
        if tenant_id filter is missing. Second line of defense.

Three-category whitelist strategy (consistent with MongoDB TenantCommandInterceptor):
    - Data-plane (explicit handling): search, count, index, bulk, get, delete, etc.
      → inject tenant_id into query/document/routing
    - Control-plane (passthrough): indices.*, cluster.*, tasks.*, ping, etc.
      → delegate to parent as-is
    - Unknown: reject
      → raise TenantIsolationViolation
"""

from typing import (
    Any,
    Collection,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from elasticsearch import AsyncElasticsearch
from elastic_transport import (
    ApiResponseMeta,
    AsyncTransport,
    HeadApiResponse,
    HttpHeaders,
    ObjectApiResponse,
)
from elastic_transport._models import DEFAULT, DefaultType, SniffOptions
from elastic_transport._otel import OpenTelemetrySpan

from core.observation.logger import get_logger
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_constants import TENANT_ID_FIELD
from core.tenants.tenant_contextvar import get_current_tenant_id

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class TenantIsolationViolation(Exception):
    """Raised when an ES operation violates tenant isolation (Layer 1)."""

    pass


class TenantGuardViolation(Exception):
    """Raised when the Guard Transport detects a tenant isolation violation (Layer 2)."""

    pass


# ============================================================================
# Section 1 — Query Utilities
# ============================================================================
#
# Pure functions to inject tenant_id filters into various ES query structures.
# Design principle: wrap at the outermost level only — never recurse into
# nested queries. This is safe because bool.filter is a pure intersection
# operation that doesn't affect inner query scoring or semantics.
# ============================================================================


def wrap_query_with_tenant(query: dict, tenant_id: str) -> dict:
    """
    Wrap any ES query with a tenant_id filter at the outermost level.

    Safe for ALL query types (bool, match, function_score, nested, knn, etc.)
    because we only add an outer bool.filter — the original query is preserved as-is.
    """
    tenant_clause = {"term": {TENANT_ID_FIELD: tenant_id}}

    if "bool" in query and len(query) == 1:
        # Merge into existing bool query's filter array
        bool_query = query["bool"]
        existing_filter = bool_query.get("filter", [])
        if isinstance(existing_filter, dict):
            existing_filter = [existing_filter]
        else:
            existing_filter = list(existing_filter)
        existing_filter.append(tenant_clause)
        bool_query["filter"] = existing_filter
        return query

    # Wrap non-bool query in a new bool
    return {"bool": {"must": [query], "filter": [tenant_clause]}}


def make_tenant_only_query(tenant_id: str) -> dict:
    """
    Create a filter-only query that matches all documents for a tenant.
    Used when the original request has no query (equivalent to match_all).
    """
    return {"bool": {"filter": [{"term": {TENANT_ID_FIELD: tenant_id}}]}}


def inject_query_body(body: Optional[dict], tenant_id: str) -> dict:
    """
    Inject tenant_id filter into a search-like request body.
    Handles: body.query, body.post_filter, body.suggest
    """
    if body is None:
        body = {}

    # suggest → blocked (term/phrase can't be isolated, completion needs mapping)
    if "suggest" in body:
        raise TenantIsolationViolation(
            "Elasticsearch suggest is blocked in tenant isolation mode. "
            "Term/Phrase suggest cannot be tenant-isolated. "
            "Completion suggest requires context filter setup. "
            "Contact platform team to enable suggest support."
        )

    # query injection
    query = body.get("query")
    if query is None:
        body["query"] = make_tenant_only_query(tenant_id)
    else:
        body["query"] = wrap_query_with_tenant(query, tenant_id)

    # post_filter injection (belt-and-suspenders)
    post_filter = body.get("post_filter")
    if post_filter is not None:
        body["post_filter"] = wrap_query_with_tenant(post_filter, tenant_id)

    return body


def inject_index_filter_body(body: Optional[dict], tenant_id: str) -> dict:
    """
    Inject tenant_id filter into body.index_filter field.
    Used by: terms_enum, field_caps
    """
    if body is None:
        body = {}

    tenant_clause = {"term": {TENANT_ID_FIELD: tenant_id}}
    index_filter = body.get("index_filter")
    if index_filter is None:
        body["index_filter"] = {"bool": {"filter": [tenant_clause]}}
    else:
        body["index_filter"] = wrap_query_with_tenant(index_filter, tenant_id)

    return body


def inject_knn_filter_body(body: Optional[dict], tenant_id: str) -> dict:
    """
    Inject tenant_id filter into body.filter field (top-level list).
    Used by: knn_search
    """
    if body is None:
        body = {}

    tenant_clause = {"term": {TENANT_ID_FIELD: tenant_id}}
    existing_filter = body.get("filter", [])
    if isinstance(existing_filter, dict):
        existing_filter = [existing_filter]
    else:
        existing_filter = list(existing_filter)
    existing_filter.append(tenant_clause)
    body["filter"] = existing_filter

    return body


def inject_msearch_body(body: Any, tenant_id: str) -> Any:
    """
    Inject tenant_id filter into each sub-request of an msearch body.

    msearch body is a list where every two elements form a pair:
    [header, body, header, body, ...]
    - Even indices (0, 2, 4, ...): search header (index, routing, etc.)
    - Odd indices (1, 3, 5, ...): search body (query, aggs, etc.)
    """
    if not isinstance(body, list):
        return body

    for i in range(0, len(body), 2):
        # Inject routing into header
        if i < len(body) and isinstance(body[i], dict):
            body[i].setdefault("routing", tenant_id)

        # Inject tenant filter into search body
        body_idx = i + 1
        if body_idx < len(body) and isinstance(body[body_idx], dict):
            body[body_idx] = inject_query_body(body[body_idx], tenant_id)

    return body


def inject_bulk_body(body: Any, tenant_id: str) -> Any:
    """
    Inject tenant_id into each action of a bulk request body.

    bulk body is NDJSON: every two elements form a pair:
    [action_meta, document, action_meta, document, ...]

    For index/create: inject tenant_id into document + routing into metadata
    For update: inject tenant_id into doc/upsert + routing into metadata
    For delete: reject (no document body to inject into)
    """
    if not isinstance(body, list):
        return body

    i = 0
    while i < len(body):
        action_meta = body[i]
        if not isinstance(action_meta, dict):
            i += 1
            continue

        action_type = next(iter(action_meta), None)
        if action_type is None:
            i += 1
            continue

        action_params = action_meta[action_type]
        if isinstance(action_params, dict):
            action_params.setdefault("routing", tenant_id)

        if action_type in ("index", "create"):
            # Next element is the document
            doc_idx = i + 1
            if doc_idx < len(body) and isinstance(body[doc_idx], dict):
                body[doc_idx][TENANT_ID_FIELD] = tenant_id
            i += 2

        elif action_type == "update":
            # Next element is the update body (doc/upsert/script)
            doc_idx = i + 1
            if doc_idx < len(body) and isinstance(body[doc_idx], dict):
                update_body = body[doc_idx]
                if "doc" in update_body and isinstance(update_body["doc"], dict):
                    update_body["doc"][TENANT_ID_FIELD] = tenant_id
                if "upsert" in update_body and isinstance(update_body["upsert"], dict):
                    update_body["upsert"][TENANT_ID_FIELD] = tenant_id
            i += 2

        elif action_type == "delete":
            # delete in bulk has no document body — cannot inject tenant filter
            raise TenantIsolationViolation(
                "bulk delete is not supported in tenant isolation mode. "
                "Use delete_by_query with tenant_id filter instead."
            )

        else:
            i += 1

    return body


def make_ids_tenant_query(doc_id: str, tenant_id: str) -> dict:
    """
    Build a query that matches a specific document ID AND tenant_id.
    Used to convert ID-based operations (get, exists, delete) to query-based ones.
    """
    return {
        "bool": {
            "filter": [
                {"ids": {"values": [doc_id]}},
                {"term": {TENANT_ID_FIELD: tenant_id}},
            ]
        }
    }


# ============================================================================
# Section 2 — Interceptor (Layer 1)
# ============================================================================
#
# TenantAwareAsyncElasticsearch: intercepts ALL ES operations via the single
# perform_request() chokepoint. No individual method overrides needed —
# perform_request() is the single method ALL 47+ AsyncElasticsearch methods
# (search, count, index, bulk, get, etc.) and all namespace clients
# (indices.*, cluster.*, tasks.*) delegate to.
#
# endpoint_id is provided by every ES SDK method call (e.g., "search", "index",
# "indices.create") and is used for OpenTelemetry tracing — stable across versions.
# ============================================================================


class TenantAwareAsyncElasticsearch(AsyncElasticsearch):
    """Tenant-aware ES client that intercepts all operations via perform_request()."""

    # ================================================================
    # Endpoint classification
    # ================================================================

    # Query endpoints: inject bool.filter(tenant_id) into body.query
    _QUERY_ENDPOINTS_ACTIVE: FrozenSet[str] = frozenset(
        {"search", "count", "delete_by_query"}
    )
    _QUERY_ENDPOINTS_BLOCKED: FrozenSet[str] = frozenset({"update_by_query"})

    # Special query endpoints: different field names for filter injection
    _SPECIAL_QUERY_ENDPOINTS_BLOCKED: FrozenSet[str] = frozenset(
        {"msearch", "knn_search", "terms_enum", "field_caps"}
    )

    # Write endpoints: inject tenant_id field into document body
    _WRITE_ENDPOINTS: FrozenSet[str] = frozenset({"index", "create"})

    # ID-based endpoints: convert to query-based equivalents
    _ID_ENDPOINTS_ACTIVE: FrozenSet[str] = frozenset({"get", "exists", "delete"})
    _ID_ENDPOINTS_BLOCKED: FrozenSet[str] = frozenset({"update"})

    # Unsupported: cannot safely inject tenant filter
    _UNSUPPORTED_ENDPOINTS: FrozenSet[str] = frozenset(
        {"search_template", "msearch_template", "rank_eval", "mget"}
    )

    # Control-plane / ops: passthrough without any injection
    _PASSTHROUGH_PREFIXES: Tuple[str, ...] = (
        "indices.",
        "cluster.",
        "tasks.",
        "nodes.",
        "snapshot.",
        "cat.",
        "ingest.",
        "ilm.",
        "security.",
        "ml.",
        "transform.",
        "watcher.",
        "xpack.",
        "async_search.",
        "slm.",
        "enrich.",
        "rollup.",
    )
    _PASSTHROUGH_ENDPOINTS: FrozenSet[str] = frozenset(
        {
            "scroll",
            "clear_scroll",
            "open_point_in_time",
            "close_point_in_time",
            "ping",
            "info",
            "explain",
            "reindex",
        }
    )

    # ================================================================
    # Main entry point
    # ================================================================

    async def perform_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Any] = None,
        endpoint_id: Optional[str] = None,
        path_parts: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        tid = self._get_tenant_id()

        # No tenant context: raise if app is ready, passthrough during startup
        if not tid:
            if get_tenant_config().app_ready:
                raise TenantIsolationViolation(
                    f"Missing tenant_id for ES endpoint '{endpoint_id}' "
                    f"({method} {path}). "
                    f"Ensure tenant context is set before data operations."
                )
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- No endpoint_id: some ES SDK methods (e.g. ping) don't set it.
        #     These are safe control-plane operations — passthrough. ---
        if endpoint_id is None:
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- Blocked endpoints (implemented but not yet enabled) ---
        if endpoint_id in self._QUERY_ENDPOINTS_BLOCKED:
            raise TenantIsolationViolation(
                f"Endpoint '{endpoint_id}' is not yet enabled in tenant isolation mode. "
                f"Implementation exists but is blocked for testing. "
                f"Contact platform team to enable."
            )
        if endpoint_id in self._SPECIAL_QUERY_ENDPOINTS_BLOCKED:
            raise TenantIsolationViolation(
                f"Endpoint '{endpoint_id}' is not yet enabled in tenant isolation mode. "
                f"Contact platform team to enable."
            )
        if endpoint_id in self._ID_ENDPOINTS_BLOCKED:
            raise TenantIsolationViolation(
                f"Endpoint '{endpoint_id}' (by ID) is not yet enabled in tenant "
                f"isolation mode. Contact platform team to enable."
            )

        # --- Unsupported endpoints (cannot safely inject) ---
        if endpoint_id in self._UNSUPPORTED_ENDPOINTS:
            raise TenantIsolationViolation(
                f"Endpoint '{endpoint_id}' is not supported in tenant isolation mode. "
                f"Cannot safely inject tenant_id filter."
            )

        # --- Control-plane / ops: passthrough ---
        if endpoint_id in self._PASSTHROUGH_ENDPOINTS:
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )
        if endpoint_id and any(
            endpoint_id.startswith(p) for p in self._PASSTHROUGH_PREFIXES
        ):
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- Query endpoints: inject tenant filter + routing (all modes) ---
        if endpoint_id in self._QUERY_ENDPOINTS_ACTIVE:
            params = self._inject_routing(params, tid)
            body = inject_query_body(body, tid)
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- Write endpoints: inject tenant_id field + routing ---
        if endpoint_id in self._WRITE_ENDPOINTS:
            params = self._inject_routing(params, tid)
            if body is None:
                raise TenantIsolationViolation(
                    f"Write operation '{endpoint_id}' has no body. "
                    f"Cannot inject tenant_id."
                )
            body = dict(body) if not isinstance(body, dict) else body
            body[TENANT_ID_FIELD] = tid
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- Bulk: per-action injection ---
        if endpoint_id == "bulk":
            body = inject_bulk_body(body, tid)
            return await super().perform_request(
                method,
                path,
                params=params,
                headers=headers,
                body=body,
                endpoint_id=endpoint_id,
                path_parts=path_parts,
            )

        # --- ID-based endpoints: convert to query-based for tenant filtering ---
        if endpoint_id in self._ID_ENDPOINTS_ACTIVE:
            path_parts = dict(path_parts or {})
            if endpoint_id == "get":
                return await self._convert_get_to_search(
                    path_parts, params, headers, tid
                )
            elif endpoint_id == "exists":
                return await self._convert_exists_to_count(
                    path_parts, params, headers, tid
                )
            elif endpoint_id == "delete":
                return await self._convert_delete_to_dbq(
                    path_parts, params, headers, tid
                )

        # --- Unknown endpoint: reject ---
        raise TenantIsolationViolation(
            f"Unknown ES endpoint '{endpoint_id}'. "
            f"Add to the appropriate set in TenantAwareAsyncElasticsearch. "
            f"This is a safety measure to prevent tenant data leakage."
        )

    # ================================================================
    # ID-based operation conversions
    # ================================================================

    async def _convert_get_to_search(
        self,
        path_parts: Dict[str, Any],
        params: Optional[Mapping[str, Any]],
        headers: Optional[Mapping[str, str]],
        tid: str,
    ) -> ObjectApiResponse:
        """
        Convert get-by-id to search with ids + tenant_id filter.

        Original: GET /{index}/_doc/{id}
        Converted: POST /{index}/_search {"query": {"bool": {"filter": [ids, tenant]}}}
        """
        index = path_parts.get("index", "")
        doc_id = path_parts.get("id", "")

        search_body = {"query": make_ids_tenant_query(doc_id, tid), "size": 1}
        search_params = {"routing": tid}

        if params:
            for key in ("_source", "_source_includes", "_source_excludes"):
                if key in params:
                    search_params[key] = params[key]

        merged_headers = dict(headers or {})
        merged_headers.setdefault("content-type", "application/json")

        response = await super().perform_request(
            "POST",
            f"/{index}/_search",
            params=search_params,
            headers=merged_headers,
            body=search_body,
            endpoint_id="search",
            path_parts={"index": index},
        )

        hits = response.body.get("hits", {}).get("hits", [])
        if hits:
            hit = hits[0]
            get_body = {
                "_index": hit.get("_index", index),
                "_id": hit.get("_id", doc_id),
                "_version": hit.get("_version", 1),
                "_seq_no": hit.get("_seq_no"),
                "_primary_term": hit.get("_primary_term"),
                "_source": hit.get("_source", {}),
                "found": True,
            }
        else:
            get_body = {"_index": index, "_id": doc_id, "found": False}

        return ObjectApiResponse(body=get_body, meta=response.meta)

    async def _convert_exists_to_count(
        self,
        path_parts: Dict[str, Any],
        params: Optional[Mapping[str, Any]],
        headers: Optional[Mapping[str, str]],
        tid: str,
    ) -> HeadApiResponse:
        """
        Convert exists-by-id to count with ids + tenant_id filter.

        Original: HEAD /{index}/_doc/{id}
        Converted: POST /{index}/_count {"query": {"bool": {"filter": [ids, tenant]}}}
        """
        index = path_parts.get("index", "")
        doc_id = path_parts.get("id", "")

        count_body = {"query": make_ids_tenant_query(doc_id, tid)}
        count_params = {"routing": tid}

        merged_headers = dict(headers or {})
        merged_headers.setdefault("content-type", "application/json")

        response = await super().perform_request(
            "POST",
            f"/{index}/_count",
            params=count_params,
            headers=merged_headers,
            body=count_body,
            endpoint_id="count",
            path_parts={"index": index},
        )

        count = response.body.get("count", 0)
        status = 200 if count > 0 else 404

        meta = ApiResponseMeta(
            status=status,
            http_version=response.meta.http_version,
            headers=response.meta.headers,
            duration=response.meta.duration,
            node=response.meta.node,
        )
        return HeadApiResponse(meta=meta)

    async def _convert_delete_to_dbq(
        self,
        path_parts: Dict[str, Any],
        params: Optional[Mapping[str, Any]],
        headers: Optional[Mapping[str, str]],
        tid: str,
    ) -> ObjectApiResponse:
        """
        Convert delete-by-id to delete_by_query with ids + tenant_id filter.

        Original: DELETE /{index}/_doc/{id}
        Converted: POST /{index}/_delete_by_query {"query": {"bool": {"filter": [ids, tenant]}}}
        """
        index = path_parts.get("index", "")
        doc_id = path_parts.get("id", "")

        dbq_body = {"query": make_ids_tenant_query(doc_id, tid)}
        dbq_params = {"routing": tid}

        merged_headers = dict(headers or {})
        merged_headers.setdefault("content-type", "application/json")

        response = await super().perform_request(
            "POST",
            f"/{index}/_delete_by_query",
            params=dbq_params,
            headers=merged_headers,
            body=dbq_body,
            endpoint_id="delete_by_query",
            path_parts={"index": index},
        )

        deleted = response.body.get("deleted", 0)
        result = "deleted" if deleted > 0 else "not_found"

        delete_body = {
            "_index": index,
            "_id": doc_id,
            "_version": 1,
            "_seq_no": 0,
            "_primary_term": 1,
            "result": result,
            "_shards": {"total": 1, "successful": 1 if deleted > 0 else 0, "failed": 0},
        }
        return ObjectApiResponse(body=delete_body, meta=response.meta)

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _get_tenant_id() -> Optional[str]:
        """Get current tenant_id from context."""
        return get_current_tenant_id()

    @staticmethod
    def _inject_routing(
        params: Optional[Mapping[str, Any]], tid: str
    ) -> Dict[str, Any]:
        """Inject routing=tenant_id into query params."""
        params = dict(params or {})
        params.setdefault("routing", tid)
        return params


# ============================================================================
# Section 3 — Guard Transport (Layer 2)
# ============================================================================
#
# Independent verification layer at the Transport level.
# Does NOT modify any data — only checks and raises alarm if tenant_id is missing.
#
# Layer 1 (TenantAwareAsyncElasticsearch) injects tenant_id at the Client level;
# this guard verifies at the Transport level — a different class, different layer.
#
# Key design: structure-based checking (inspects body content, not endpoint names).
# This automatically covers any new endpoint that Layer 1 starts supporting,
# without needing to sync endpoint name lists between layers.
#
# At this level, body is still a structured dict — serialization to bytes happens
# AFTER this check (in the parent's perform_request).
# ============================================================================


class TenantGuardTransport(AsyncTransport):
    """
    Independent verification layer that checks tenant_id presence in ES requests.

    Does NOT modify anything — only raises TenantGuardViolation on violations.

    Verification is structure-based (checks body content, not endpoint names):
    - body has "query" → query must contain tenant_id filter
    - body has "index_filter" → must contain tenant_id filter
    - body has "filter" as list → must contain tenant_id
    - body has TENANT_ID_FIELD → must match current tenant
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._violation_count = 0

    @property
    def violation_count(self) -> int:
        """Number of detected violations since creation."""
        return self._violation_count

    async def perform_request(
        self,
        method: str,
        target: str,
        *,
        body: Optional[Any] = None,
        headers: Union[Mapping[str, Any], DefaultType] = DEFAULT,
        max_retries: Union[int, DefaultType] = DEFAULT,
        retry_on_status: Union[Collection[int], DefaultType] = DEFAULT,
        retry_on_timeout: Union[bool, DefaultType] = DEFAULT,
        request_timeout: Union[Optional[float], DefaultType] = DEFAULT,
        client_meta: Union[Tuple[Tuple[str, str], ...], DefaultType] = DEFAULT,
        otel_span: Union[OpenTelemetrySpan, DefaultType] = DEFAULT,
    ) -> Any:
        """Override to verify tenant isolation before parent serializes body."""
        self._verify_tenant_isolation(method, target, body)

        return await super().perform_request(
            method,
            target,
            body=body,
            headers=headers,
            max_retries=max_retries,
            retry_on_status=retry_on_status,
            retry_on_timeout=retry_on_timeout,
            request_timeout=request_timeout,
            client_meta=client_meta,
            otel_span=otel_span,
        )

    def _verify_tenant_isolation(
        self, method: str, target: str, body: Optional[Any]
    ) -> None:
        """
        Structure-based verification — checks body content, not endpoint names.

        This makes the Guard automatically cover any new endpoint that Layer 1
        starts supporting, without needing to sync endpoint lists.
        """
        tid = self._get_tenant_id()
        if not tid:
            if get_tenant_config().app_ready:
                self._raise_violation(
                    target, "missing tenant_id — ensure tenant context is set"
                )
            return
        if body is None:
            return
        if not isinstance(body, dict):
            return

        # 1. body has "query" → query must contain tenant_id filter
        if "query" in body:
            if not self._query_has_tenant(body["query"], tid):
                self._raise_violation(target, "query missing tenant_id filter")

        # 2. body has "index_filter" → must contain tenant_id filter
        if "index_filter" in body:
            if not self._query_has_tenant(body["index_filter"], tid):
                self._raise_violation(target, "index_filter missing tenant_id filter")

        # 3. body has "filter" as top-level list → check tenant in filter list
        if "filter" in body and isinstance(body["filter"], list):
            if not any(
                f.get("term", {}).get(TENANT_ID_FIELD) == tid
                for f in body["filter"]
                if isinstance(f, dict)
            ):
                self._raise_violation(target, "filter list missing tenant_id")

        # 4. body has "tenant_id" field → must match current tenant
        if TENANT_ID_FIELD in body:
            if body[TENANT_ID_FIELD] != tid:
                self._raise_violation(
                    target, f"tenant_id mismatch: {body['tenant_id']} != {tid}"
                )

    def _raise_violation(self, target: str, reason: str) -> None:
        """Log and raise a tenant isolation violation."""
        self._violation_count += 1
        msg = (
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!  ES TENANT ISOLATION VIOLATION — DATA LEAK RISK         !!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"!!  Target:     {target}\n"
            f"!!  Violation:  {reason}\n"
            f"!!  Cause:      Layer 1 interceptor was bypassed or has a bug\n"
            f"!!  Count:      #{self._violation_count}\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        logger.error(msg)
        raise TenantGuardViolation(msg)

    @staticmethod
    def _get_tenant_id() -> Optional[str]:
        """Get current tenant_id from context."""
        return get_current_tenant_id()

    @staticmethod
    def _query_has_tenant(query: Any, expected_tid: str) -> bool:
        """
        Check if a query dict contains tenant_id filter in bool.filter.

        Checks the outermost bool.filter array for a term query matching tenant_id.
        This is where Layer 1 injects the filter, so if it's missing here,
        the injection was bypassed.
        """
        if not isinstance(query, dict):
            return False
        if "bool" not in query:
            return False

        bool_q = query["bool"]
        filters = bool_q.get("filter", [])
        if isinstance(filters, dict):
            filters = [filters]

        for f in filters:
            if not isinstance(f, dict):
                continue
            term = f.get("term")
            if isinstance(term, dict) and term.get(TENANT_ID_FIELD) == expected_tid:
                return True

        return False
