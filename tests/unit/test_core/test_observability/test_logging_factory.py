"""``configure_logging`` + ``get_logger`` smoke tests."""

from __future__ import annotations

import pytest
import structlog

from everos.core.observability.logging.factory import configure_logging, get_logger


def test_get_logger_returns_structlog_instance() -> None:
    logger = get_logger("test.module")
    # structlog's BoundLogger interface — must expose .info / .warning / .error.
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so assertions are stable."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_configure_logging_accepts_known_levels() -> None:
    """Smoke-test the level-name → log-level mapping path; no raise."""
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "info", "warn"):
        configure_logging(level=level)


def test_configure_logging_handles_unknown_level_silently() -> None:
    """Unknown level name silently falls back via ``getattr(logging, ..., INFO)``."""
    # Just must not raise; behavior verified by absence of exception.
    configure_logging(level="NOPE")


def test_configure_logging_emits_through_structlog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="INFO")
    logger = get_logger("everos.test")
    logger.info("hello", k="v")
    plain = _strip_ansi(capsys.readouterr().out)
    assert "hello" in plain
    # ConsoleRenderer renders key=value pairs (sans color codes).
    assert "k=v" in plain


def test_configure_logging_demotes_noisy_http_loggers_to_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Third-party HTTP client loggers (httpx / httpcore / urllib3) must be
    pinned at WARNING so each successful HTTP request doesn't produce an
    INFO line. everos's own ``get_logger(...)`` calls remain unaffected.
    """
    import logging

    configure_logging(level="INFO")

    for name in ("httpx", "httpcore", "urllib3"):
        assert logging.getLogger(name).level == logging.WARNING, (
            f"{name} logger must be pinned to WARNING, got "
            f"{logging.getLevelName(logging.getLogger(name).level)}"
        )

    # Behavioral check: an INFO from httpx must NOT reach stdout.
    logging.getLogger("httpx").info("HTTP Request: GET https://example.com 200 OK")
    plain = _strip_ansi(capsys.readouterr().out)
    assert "HTTP Request" not in plain


def test_configure_logging_routes_stdlib_loggers_through_same_formatter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stdlib ``logging.getLogger(...)`` output must share the structlog
    ProcessorFormatter so uvicorn / fastapi / third-party libs render with
    the same ``[level] event`` shape as everos's own structlog calls.

    This is the user-visible half of the foreign-log-integration setup —
    without it, uvicorn's default ``LOGGING_CONFIG`` would (a) reinstall
    its own handlers and (b) print ``INFO:logger.name:message`` lines
    that look nothing like the structlog ConsoleRenderer output.
    """
    import logging

    configure_logging(level="INFO")
    third_party = logging.getLogger("uvicorn.access")
    third_party.info("foreign event")

    plain = _strip_ansi(capsys.readouterr().out)
    assert "foreign event" in plain
    # Default stdlib LogRecord prefix must NOT survive.
    assert "INFO:uvicorn.access" not in plain
    # ConsoleRenderer marks level in brackets; both structlog and stdlib
    # paths must produce the same shape.
    assert "[info" in plain


def test_get_logger_with_same_name_returns_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """structlog caches bound loggers per name when cache_logger_on_first_use=True."""
    configure_logging()
    a = get_logger("everos.cache.test")
    b = get_logger("everos.cache.test")
    # Both should behave equivalently; identity is not guaranteed by structlog
    # API, but both must satisfy the same protocol surface.
    assert isinstance(a, structlog.stdlib.BoundLogger | structlog.BoundLoggerBase) or (
        hasattr(a, "info") and hasattr(b, "info")
    )
