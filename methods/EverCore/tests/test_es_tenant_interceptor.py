"""
Test: ES Tenant Field Interceptor

Unit tests for TenantAwareAsyncElasticsearch (Layer 1) and TenantGuardTransport (Layer 2).
Tests verify that tenant_id is correctly injected into all data-plane operations.

These are pure unit tests — no real ES connection needed.
We mock super().perform_request() to capture the modified body/params.

Run:
    PYTHONPATH=src pytest tests/test_es_tenant_interceptor.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from elastic_transport import ApiResponseMeta, HttpHeaders, ObjectApiResponse

from core.tenants.tenant_constants import TENANT_ID_FIELD

# ==================== Test Constants ====================

TEST_TENANT_ID = "test_tenant_001"
TEST_INDEX = "test_memories"


# ==================== Fixtures ====================


def _make_meta(status: int = 200) -> ApiResponseMeta:
    """Create a minimal ApiResponseMeta for testing."""
    return ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders({"x-elastic-product": "Elasticsearch"}),
        duration=0.01,
        node=MagicMock(),
    )


def _make_search_response(hits: list, meta: Optional[ApiResponseMeta] = None):
    """Create a mock search response."""
    body = {"hits": {"total": {"value": len(hits)}, "hits": hits}}
    return ObjectApiResponse(body=body, meta=meta or _make_meta())


def _make_count_response(count: int, meta: Optional[ApiResponseMeta] = None):
    """Create a mock count response."""
    body = {"count": count}
    return ObjectApiResponse(body=body, meta=meta or _make_meta())


def _make_dbq_response(deleted: int, meta: Optional[ApiResponseMeta] = None):
    """Create a mock delete_by_query response."""
    body = {"took": 10, "deleted": deleted, "total": deleted}
    return ObjectApiResponse(body=body, meta=meta or _make_meta())


@pytest.fixture
def mock_tenant_context():
    """Mock tenant context to return TEST_TENANT_ID in shared mode."""
    mock_tenant_info = MagicMock()
    mock_tenant_info.is_shared_mode = True

    with (
        patch(
            "core.tenants.tenantize.oxm.es.tenant_field_es_interceptor.get_current_tenant_id",
            return_value=TEST_TENANT_ID,
        ),
        patch(
            "core.tenants.tenantize.oxm.es.tenant_field_es_interceptor.get_current_tenant",
            return_value=mock_tenant_info,
        ),
    ):
        yield


@pytest.fixture
def mock_no_tenant_context():
    """Mock no tenant context (tenant_id returns None)."""
    with patch(
        "core.tenants.tenantize.oxm.es.tenant_field_es_interceptor.get_current_tenant_id",
        return_value=None,
    ):
        yield


# ==================== Query Utility Tests ====================


class TestTenantQueryUtils:
    """Test the query injection utility functions."""

    def test_wrap_query_with_tenant_non_bool(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            wrap_query_with_tenant,
        )

        query = {"match": {"content": "hello"}}
        result = wrap_query_with_tenant(query, TEST_TENANT_ID)

        assert "bool" in result
        assert result["bool"]["must"] == [{"match": {"content": "hello"}}]
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in result["bool"]["filter"]

    def test_wrap_query_with_tenant_existing_bool(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            wrap_query_with_tenant,
        )

        query = {"bool": {"must": [{"match": {"content": "hello"}}]}}
        result = wrap_query_with_tenant(query, TEST_TENANT_ID)

        # Should merge into existing bool, not wrap again
        assert result["bool"]["must"] == [{"match": {"content": "hello"}}]
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in result["bool"]["filter"]

    def test_wrap_query_with_tenant_existing_bool_with_filter(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            wrap_query_with_tenant,
        )

        query = {
            "bool": {
                "must": [{"match": {"content": "hello"}}],
                "filter": [{"term": {"user_id": "u001"}}],
            }
        }
        result = wrap_query_with_tenant(query, TEST_TENANT_ID)

        # Original filter preserved + tenant filter appended
        filters = result["bool"]["filter"]
        assert {"term": {"user_id": "u001"}} in filters
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in filters
        assert len(filters) == 2

    def test_wrap_query_with_tenant_filter_as_dict(self):
        """When existing filter is a dict (not list), should convert to list."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            wrap_query_with_tenant,
        )

        query = {"bool": {"filter": {"term": {"status": "active"}}}}
        result = wrap_query_with_tenant(query, TEST_TENANT_ID)

        filters = result["bool"]["filter"]
        assert isinstance(filters, list)
        assert {"term": {"status": "active"}} in filters
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in filters

    def test_make_tenant_only_query(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            make_tenant_only_query,
        )

        result = make_tenant_only_query(TEST_TENANT_ID)
        assert result == {
            "bool": {"filter": [{"term": {TENANT_ID_FIELD: TEST_TENANT_ID}}]}
        }

    def test_inject_query_body_none(self):
        """body=None should create a filter-only query, never skip."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )

        result = inject_query_body(None, TEST_TENANT_ID)
        assert result["query"]["bool"]["filter"] == [
            {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}}
        ]

    def test_inject_query_body_empty(self):
        """Empty body should create a filter-only query."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )

        result = inject_query_body({}, TEST_TENANT_ID)
        assert result["query"]["bool"]["filter"] == [
            {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}}
        ]

    def test_inject_query_body_no_query(self):
        """body with size but no query should add tenant filter."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )

        result = inject_query_body({"size": 10}, TEST_TENANT_ID)
        assert "query" in result
        assert result["size"] == 10

    def test_inject_query_body_with_post_filter(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )

        body = {
            "query": {"match_all": {}},
            "post_filter": {"term": {"status": "active"}},
        }
        result = inject_query_body(body, TEST_TENANT_ID)

        # Both query and post_filter should have tenant filter
        assert "bool" in result["query"]
        assert "bool" in result["post_filter"]

    def test_inject_query_body_with_suggest_raises(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantIsolationViolation,
        )

        body = {
            "query": {"match_all": {}},
            "suggest": {"my_suggest": {"text": "test", "term": {"field": "content"}}},
        }
        with pytest.raises(TenantIsolationViolation, match="suggest"):
            inject_query_body(body, TEST_TENANT_ID)

    def test_inject_query_body_preserves_aggs(self):
        """Aggregations should be left untouched."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_query_body,
        )

        body = {
            "query": {"match": {"content": "test"}},
            "aggs": {"by_user": {"terms": {"field": "user_id"}}},
        }
        result = inject_query_body(body, TEST_TENANT_ID)

        # aggs preserved as-is
        assert result["aggs"] == {"by_user": {"terms": {"field": "user_id"}}}

    def test_make_ids_tenant_query(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            make_ids_tenant_query,
        )

        result = make_ids_tenant_query("doc_123", TEST_TENANT_ID)
        assert result == {
            "bool": {
                "filter": [
                    {"ids": {"values": ["doc_123"]}},
                    {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}},
                ]
            }
        }


# ==================== Bulk Injection Tests ====================


class TestBulkInjection:
    def test_inject_bulk_index(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_bulk_body,
        )

        body = [
            {"index": {"_index": "memories", "_id": "1"}},
            {"content": "hello", "user_id": "u1"},
            {"index": {"_index": "memories", "_id": "2"}},
            {"content": "world", "user_id": "u2"},
        ]
        result = inject_bulk_body(body, TEST_TENANT_ID)

        # Check routing injected into action metadata
        assert result[0]["index"]["routing"] == TEST_TENANT_ID
        assert result[2]["index"]["routing"] == TEST_TENANT_ID

        # Check tenant_id injected into documents
        assert result[1]["tenant_id"] == TEST_TENANT_ID
        assert result[3]["tenant_id"] == TEST_TENANT_ID

    def test_inject_bulk_create(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_bulk_body,
        )

        body = [{"create": {"_index": "memories", "_id": "1"}}, {"content": "hello"}]
        result = inject_bulk_body(body, TEST_TENANT_ID)

        assert result[0]["create"]["routing"] == TEST_TENANT_ID
        assert result[1]["tenant_id"] == TEST_TENANT_ID

    def test_inject_bulk_update(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_bulk_body,
        )

        body = [
            {"update": {"_index": "memories", "_id": "1"}},
            {"doc": {"content": "updated"}, "upsert": {"content": "new"}},
        ]
        result = inject_bulk_body(body, TEST_TENANT_ID)

        assert result[0]["update"]["routing"] == TEST_TENANT_ID
        assert result[1]["doc"]["tenant_id"] == TEST_TENANT_ID
        assert result[1]["upsert"]["tenant_id"] == TEST_TENANT_ID

    def test_inject_bulk_delete_raises(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            inject_bulk_body,
        )
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantIsolationViolation,
        )

        body = [{"delete": {"_index": "memories", "_id": "1"}}]
        with pytest.raises(TenantIsolationViolation, match="bulk delete"):
            inject_bulk_body(body, TEST_TENANT_ID)


# ==================== Interceptor (Layer 1) Tests ====================


class TestTenantAwareAsyncElasticsearch:
    """Test the perform_request override."""

    @pytest.mark.asyncio
    async def test_search_injects_tenant_filter(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = kwargs.get("body")
            captured["params"] = kwargs.get("params")
            return _make_search_response([])

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            await client.perform_request(
                "POST",
                "/test/_search",
                body={"query": {"match": {"content": "hello"}}},
                endpoint_id="search",
                path_parts={"index": "test"},
            )

        # Verify tenant filter injected
        query = captured["body"]["query"]
        assert "bool" in query
        filters = query["bool"]["filter"]
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in filters

        # Verify routing injected
        assert captured["params"]["routing"] == TEST_TENANT_ID

    @pytest.mark.asyncio
    async def test_search_body_none_creates_filter(self, mock_tenant_context):
        """body=None must not be skipped — should create tenant filter query."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["body"] = kwargs.get("body")
            return _make_search_response([])

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            await client.perform_request(
                "POST",
                "/test/_search",
                body=None,
                endpoint_id="search",
                path_parts={"index": "test"},
            )

        # body should have been created with tenant filter
        assert captured["body"] is not None
        assert "query" in captured["body"]
        filters = captured["body"]["query"]["bool"]["filter"]
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in filters

    @pytest.mark.asyncio
    async def test_index_injects_tenant_id(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["body"] = kwargs.get("body")
            captured["params"] = kwargs.get("params")
            return ObjectApiResponse(
                body={"_id": "1", "result": "created"}, meta=_make_meta()
            )

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            await client.perform_request(
                "PUT",
                "/test/_doc/1",
                body={"content": "hello", "user_id": "u1"},
                endpoint_id="index",
                path_parts={"index": "test", "id": "1"},
            )

        assert captured["body"]["tenant_id"] == TEST_TENANT_ID
        assert captured["params"]["routing"] == TEST_TENANT_ID

    @pytest.mark.asyncio
    async def test_index_body_none_raises(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
            TenantIsolationViolation,
        )

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with pytest.raises(TenantIsolationViolation, match="no body"):
            await client.perform_request(
                "PUT",
                "/test/_doc/1",
                body=None,
                endpoint_id="index",
                path_parts={"index": "test", "id": "1"},
            )

    @pytest.mark.asyncio
    async def test_get_converts_to_search(self, mock_tenant_context):
        """get by ID should be converted to search with ids + tenant filter."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = kwargs.get("body")
            captured["endpoint_id"] = kwargs.get("endpoint_id")
            return _make_search_response(
                [
                    {
                        "_index": TEST_INDEX,
                        "_id": "doc_123",
                        "_version": 1,
                        "_source": {"content": "hello", "tenant_id": TEST_TENANT_ID},
                    }
                ]
            )

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            response = await client.perform_request(
                "GET",
                f"/{TEST_INDEX}/_doc/doc_123",
                endpoint_id="get",
                path_parts={"index": TEST_INDEX, "id": "doc_123"},
            )

        # Should have been converted to search
        assert captured["method"] == "POST"
        assert "_search" in captured["path"]
        assert captured["endpoint_id"] == "search"

        # Query should have ids + tenant filter
        query_filters = captured["body"]["query"]["bool"]["filter"]
        assert {"ids": {"values": ["doc_123"]}} in query_filters
        assert {"term": {TENANT_ID_FIELD: TEST_TENANT_ID}} in query_filters

        # Response should be in get format
        assert response.body["found"] is True
        assert response.body["_id"] == "doc_123"
        assert response.body["_source"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_get_not_found_returns_found_false(self, mock_tenant_context):
        """get for non-existent or other-tenant doc should return found=False."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        async def mock_super_perform_request(method, path, **kwargs):
            return _make_search_response([])  # No hits

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            response = await client.perform_request(
                "GET",
                f"/{TEST_INDEX}/_doc/doc_other_tenant",
                endpoint_id="get",
                path_parts={"index": TEST_INDEX, "id": "doc_other_tenant"},
            )

        assert response.body["found"] is False

    @pytest.mark.asyncio
    async def test_delete_converts_to_dbq(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["method"] = method
            captured["endpoint_id"] = kwargs.get("endpoint_id")
            captured["body"] = kwargs.get("body")
            return _make_dbq_response(1)

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            response = await client.perform_request(
                "DELETE",
                f"/{TEST_INDEX}/_doc/doc_123",
                endpoint_id="delete",
                path_parts={"index": TEST_INDEX, "id": "doc_123"},
            )

        assert captured["method"] == "POST"
        assert captured["endpoint_id"] == "delete_by_query"
        assert response.body["result"] == "deleted"

    @pytest.mark.asyncio
    async def test_unknown_endpoint_raises(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
            TenantIsolationViolation,
        )

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with pytest.raises(TenantIsolationViolation, match="Unknown"):
            await client.perform_request(
                "POST", "/test/_some_new_api", endpoint_id="some_new_api", path_parts={}
            )

    @pytest.mark.asyncio
    async def test_blocked_endpoint_raises(self, mock_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
            TenantIsolationViolation,
        )

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with pytest.raises(TenantIsolationViolation, match="not yet enabled"):
            await client.perform_request(
                "POST",
                "/test/_search",
                body={"query": {"match_all": {}}},
                endpoint_id="update_by_query",
                path_parts={"index": "test"},
            )

    @pytest.mark.asyncio
    async def test_passthrough_when_no_tenant_context(self, mock_no_tenant_context):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["body"] = kwargs.get("body")
            return _make_search_response([])

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            await client.perform_request(
                "POST",
                "/test/_search",
                body={"query": {"match_all": {}}},
                endpoint_id="search",
                path_parts={"index": "test"},
            )

        # Without tenant context, body should be unmodified
        assert captured["body"] == {"query": {"match_all": {}}}

    @pytest.mark.asyncio
    async def test_indices_passthrough(self, mock_tenant_context):
        """Control-plane endpoints should pass through without injection."""
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantAwareAsyncElasticsearch,
        )

        captured = {}

        async def mock_super_perform_request(method, path, **kwargs):
            captured["body"] = kwargs.get("body")
            return ObjectApiResponse(body={"acknowledged": True}, meta=_make_meta())

        client = TenantAwareAsyncElasticsearch.__new__(TenantAwareAsyncElasticsearch)

        with patch.object(
            TenantAwareAsyncElasticsearch.__bases__[0],
            "perform_request",
            side_effect=mock_super_perform_request,
        ):
            await client.perform_request(
                "PUT",
                "/test_index",
                body={"settings": {"number_of_shards": 1}},
                endpoint_id="indices.create",
                path_parts={"index": "test_index"},
            )

        # Body should be unmodified (no tenant filter injected)
        assert captured["body"] == {"settings": {"number_of_shards": 1}}


# ==================== Guard Transport (Layer 2) Tests ====================


class TestTenantGuardTransport:
    """Test the structure-based verification."""

    def test_query_has_tenant_positive(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantGuardTransport,
        )

        query = {
            "bool": {
                "must": [{"match": {"content": "hello"}}],
                "filter": [{"term": {TENANT_ID_FIELD: TEST_TENANT_ID}}],
            }
        }
        assert TenantGuardTransport._query_has_tenant(query, TEST_TENANT_ID) is True

    def test_query_has_tenant_negative(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantGuardTransport,
        )

        query = {"bool": {"must": [{"match": {"content": "hello"}}]}}
        assert TenantGuardTransport._query_has_tenant(query, TEST_TENANT_ID) is False

    def test_query_has_tenant_wrong_id(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantGuardTransport,
        )

        query = {"bool": {"filter": [{"term": {"tenant_id": "wrong_tenant"}}]}}
        assert TenantGuardTransport._query_has_tenant(query, TEST_TENANT_ID) is False

    def test_query_has_tenant_non_bool(self):
        from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
            TenantGuardTransport,
        )

        query = {"match_all": {}}
        assert TenantGuardTransport._query_has_tenant(query, TEST_TENANT_ID) is False
