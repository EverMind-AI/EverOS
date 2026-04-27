"""End-to-end integration test for request_id auto-injection in logs.

Verifies the complete chain:
  Request (X-Request-Id header)
  -> AppLogicMiddleware (extracts request_id, sets ContextVar)
  -> RequestIdFilter (reads ContextVar, injects into LogRecord)
  -> Log output contains [request_id]

Does NOT require external services (MongoDB, ES, Milvus, Redis).
"""

import io
import logging
import re
from unittest.mock import patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from core.observation.logger import RequestIdFilter, get_logger
from core.middleware.app_logic_middleware import AppLogicMiddleware
from core.request.app_logic_provider import AppLogicProviderImpl


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with AppLogicMiddleware.

    Patches DI resolution so no container setup is needed.
    """
    app = FastAPI()

    with patch(
        "core.middleware.app_logic_middleware.get_bean_by_type",
        return_value=AppLogicProviderImpl(),
    ):
        app.add_middleware(AppLogicMiddleware)

    test_logger = get_logger("test.e2e.request_id")

    @app.get("/test-log")
    async def test_log_endpoint():
        test_logger.info("Integration test log message")
        return {"status": "ok"}

    return app


class TestRequestIdE2EIntegration:
    """End-to-end: request -> middleware -> ContextVar -> Filter -> log output."""

    def setup_method(self):
        """Attach a capturing handler to root logger."""
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - [%(request_id)s] - %(name)s - %(message)s"
            )
        )
        self.handler.addFilter(RequestIdFilter())
        logging.root.addHandler(self.handler)

    def teardown_method(self):
        """Remove the capturing handler."""
        logging.root.removeHandler(self.handler)

    def test_custom_request_id_from_header_appears_in_log(self):
        """X-Request-Id header value should appear in log output."""
        app = _create_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/test-log", headers={"X-Request-Id": "e2e-test-12345"})

        assert response.status_code == 200
        log_output = self.stream.getvalue()
        assert (
            "[e2e-test-12345]" in log_output
        ), f"Expected [e2e-test-12345] in log output, got:\n{log_output}"
        assert "Integration test log message" in log_output

    def test_auto_generated_uuid_when_no_header(self):
        """Without X-Request-Id header, a UUID should be auto-generated (not '-')."""
        app = _create_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/test-log")

        assert response.status_code == 200
        log_output = self.stream.getvalue()
        # Should NOT be "-" (that's for no-context scenarios like startup)
        assert (
            "[-]" not in log_output
            or "Integration test log message" not in log_output.split("[-]")[-1]
        ), f"Expected auto-generated UUID, not '-', in log output:\n{log_output}"
        # Should contain a UUID pattern
        match = re.search(
            r"\[([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\]",
            log_output,
        )
        assert match is not None, f"Expected UUID in log output, got:\n{log_output}"

    def test_lowercase_x_request_id_header(self):
        """x-request-id (lowercase) should also be extracted correctly."""
        app = _create_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get(
            "/test-log", headers={"x-request-id": "lowercase-header-test"}
        )

        assert response.status_code == 200
        log_output = self.stream.getvalue()
        assert (
            "[lowercase-header-test]" in log_output
        ), f"Expected [lowercase-header-test] in log output, got:\n{log_output}"

    def test_multiple_requests_have_isolated_request_ids(self):
        """Consecutive requests should each get their own request_id in logs."""
        app = _create_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        client.get("/test-log", headers={"X-Request-Id": "req-AAA"})
        client.get("/test-log", headers={"X-Request-Id": "req-BBB"})

        log_output = self.stream.getvalue()
        lines = log_output.strip().split("\n")

        # Find lines containing our test messages
        test_lines = [l for l in lines if "Integration test log message" in l]
        assert (
            len(test_lines) >= 2
        ), f"Expected >= 2 test log lines, got {len(test_lines)}"

        assert (
            "[req-AAA]" in test_lines[0]
        ), f"First request should have req-AAA: {test_lines[0]}"
        assert (
            "[req-BBB]" in test_lines[1]
        ), f"Second request should have req-BBB: {test_lines[1]}"
        # Verify no cross-contamination
        assert "[req-BBB]" not in test_lines[0]
        assert "[req-AAA]" not in test_lines[1]
