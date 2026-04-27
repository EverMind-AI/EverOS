"""Unit tests for RequestIdFilter.

Tests cover:
- request_id injected from app_info_context
- Fallback to "-" when no context set
- Fallback to "-" when app_info has no request_id key
- Filter always returns True (never suppresses records)
- Log format includes [request_id] section
- End-to-end log output verification
"""

import io
import logging
from unittest.mock import patch

from core.observation.logger import RequestIdFilter


class TestRequestIdFilter:
    """Test RequestIdFilter injects request_id into LogRecord."""

    def _make_record(self) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

    def test_injects_request_id_from_context(self):
        """When app_info has request_id, it should be injected."""
        f = RequestIdFilter()
        record = self._make_record()
        with patch(
            "core.observation.logger.get_current_app_info",
            return_value={"request_id": "abc-123"},
        ):
            result = f.filter(record)
        assert result is True
        assert record.request_id == "abc-123"

    def test_fallback_when_no_context(self):
        """When app_info is None, request_id should be '-'."""
        f = RequestIdFilter()
        record = self._make_record()
        with patch("core.observation.logger.get_current_app_info", return_value=None):
            f.filter(record)
        assert record.request_id == "-"

    def test_fallback_when_no_request_id_key(self):
        """When app_info exists but has no request_id, should be '-'."""
        f = RequestIdFilter()
        record = self._make_record()
        with patch(
            "core.observation.logger.get_current_app_info",
            return_value={"other_key": "value"},
        ):
            f.filter(record)
        assert record.request_id == "-"

    def test_always_returns_true(self):
        """Filter should never suppress log records."""
        f = RequestIdFilter()
        record = self._make_record()
        with patch("core.observation.logger.get_current_app_info", return_value=None):
            assert f.filter(record) is True


class TestLogFormatIntegration:
    """Test that log format and filter are properly registered."""

    def test_root_handler_has_request_id_in_format(self):
        """Root handler format should contain %(request_id)s."""
        has_format = any(
            h.formatter and "%(request_id)s" in h.formatter._fmt
            for h in logging.root.handlers
        )
        assert has_format, "No root handler has %(request_id)s in format"

    def test_root_handler_has_request_id_filter(self):
        """At least one root handler should have RequestIdFilter attached."""
        has_filter = any(
            RequestIdFilter in [type(f) for f in h.filters]
            for h in logging.root.handlers
        )
        assert has_filter, "No root handler has RequestIdFilter attached"

    def test_end_to_end_log_output_with_request_id(self):
        """Log output should contain [request_id] when context is set."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter("%(levelname)s - [%(request_id)s] - %(message)s")
        )
        handler.addFilter(RequestIdFilter())

        test_logger = logging.getLogger("test.e2e")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        try:
            with patch(
                "core.observation.logger.get_current_app_info",
                return_value={"request_id": "e2e-test-id"},
            ):
                test_logger.info("hello")

            output = stream.getvalue()
            assert "[e2e-test-id]" in output
            assert "hello" in output
        finally:
            test_logger.removeHandler(handler)

    def test_end_to_end_log_output_without_context(self):
        """Log output should contain [-] when no context is set."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter("%(levelname)s - [%(request_id)s] - %(message)s")
        )
        handler.addFilter(RequestIdFilter())

        test_logger = logging.getLogger("test.e2e.no_ctx")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        try:
            with patch(
                "core.observation.logger.get_current_app_info", return_value=None
            ):
                test_logger.info("startup")

            output = stream.getvalue()
            assert "[-]" in output
            assert "startup" in output
        finally:
            test_logger.removeHandler(handler)
