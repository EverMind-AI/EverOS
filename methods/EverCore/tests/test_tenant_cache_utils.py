"""
Test: TenantContextMissingError propagation in tenant_cache_utils

Verifies that when app is ready but tenant context is missing,
TenantContextMissingError is raised instead of silently falling back.

Run:
    PYTHONPATH=src uv run pytest tests/test_tenant_cache_utils.py -v
"""

from unittest.mock import patch, MagicMock

import pytest

from core.constants.exceptions import CriticalError
from core.tenants.tenant_models import TenantPatchKey
from core.tenants.tenantize.tenant_cache_utils import (
    get_or_compute_tenant_cache,
    TenantContextMissingError,
)


@pytest.fixture
def mock_app_ready():
    """Mock tenant config with app_ready=True and no tenant context."""
    config = MagicMock()
    config.app_ready = True
    with (
        patch(
            "core.tenants.tenantize.tenant_cache_utils.get_tenant_config",
            return_value=config,
        ),
        patch(
            "core.tenants.tenantize.tenant_cache_utils.get_current_tenant",
            return_value=None,
        ),
    ):
        yield config


@pytest.fixture
def mock_app_not_ready():
    """Mock tenant config with app_ready=False and no tenant context."""
    config = MagicMock()
    config.app_ready = False
    with (
        patch(
            "core.tenants.tenantize.tenant_cache_utils.get_tenant_config",
            return_value=config,
        ),
        patch(
            "core.tenants.tenantize.tenant_cache_utils.get_current_tenant",
            return_value=None,
        ),
    ):
        yield config


class TestTenantContextMissingError:
    """Test that strict tenant check raises TenantContextMissingError after app startup."""

    def test_app_ready_no_tenant_raises_error(self, mock_app_ready):
        """When app is ready and tenant context is missing, should raise even with fallback."""
        with pytest.raises(
            TenantContextMissingError, match="Strict tenant check failed"
        ):
            get_or_compute_tenant_cache(
                patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
                compute_func=lambda: "computed",
                fallback="default",
                cache_description="test cache",
            )

    def test_app_ready_no_tenant_raises_error_callable_fallback(self, mock_app_ready):
        """Callable fallback should not be invoked when strict check fails."""
        fallback_called = False

        def fallback_func():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_value"

        with pytest.raises(TenantContextMissingError):
            get_or_compute_tenant_cache(
                patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
                compute_func=lambda: "computed",
                fallback=fallback_func,
                cache_description="test cache",
            )

        assert (
            not fallback_called
        ), "Fallback should not be called when strict check fails"

    def test_app_not_ready_uses_fallback(self, mock_app_not_ready):
        """During startup (app not ready), should use fallback instead of raising."""
        result = get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
            compute_func=lambda: "computed",
            fallback="default",
            cache_description="test cache",
        )
        assert result == "default"

    def test_app_not_ready_no_fallback_raises_runtime_error(self, mock_app_not_ready):
        """During startup with no fallback, should raise RuntimeError (not TenantContextMissingError)."""
        with pytest.raises(RuntimeError, match="no fallback provided"):
            get_or_compute_tenant_cache(
                patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
                compute_func=lambda: "computed",
                fallback=None,
                cache_description="test cache",
            )

    def test_error_inherits_critical_error(self):
        """TenantContextMissingError should be a CriticalError (and thus Exception)."""
        assert issubclass(TenantContextMissingError, CriticalError)
        assert issubclass(TenantContextMissingError, Exception)

    def test_error_not_swallowed_by_except_exception_in_cache_func(
        self, mock_app_ready
    ):
        """The outer except Exception in get_or_compute_tenant_cache should not swallow it."""
        with pytest.raises(TenantContextMissingError):
            get_or_compute_tenant_cache(
                patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
                compute_func=lambda: "computed",
                fallback="default",
                cache_description="test cache",
            )


class TestReraiseGatherCriticalErrors:
    """Test that reraise_critical_errors works with asyncio.gather patterns."""

    def test_reraise_critical_error(self):
        """CriticalError in gather results should be re-raised."""
        from common_utils.async_utils import reraise_critical_errors

        error = TenantContextMissingError("tenant missing")
        results = ["ok", error, "also ok"]
        with pytest.raises(TenantContextMissingError, match="tenant missing"):
            reraise_critical_errors(results)

    def test_regular_exceptions_not_reraised(self):
        """Regular Exception in gather results should NOT be re-raised."""
        from common_utils.async_utils import reraise_critical_errors

        results = ["ok", ValueError("some error"), "also ok"]
        reraise_critical_errors(results)  # Should not raise

    def test_gather_isinstance_exception_still_matches(self):
        """isinstance(error, Exception) should be True — CriticalError IS an Exception."""
        error = TenantContextMissingError("test")
        assert isinstance(error, Exception)
        assert isinstance(error, CriticalError)
