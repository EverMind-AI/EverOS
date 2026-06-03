"""Shared pytest fixtures.

Cache invalidation:
    ``load_settings`` (and the timezone helper that reads it) are
    ``functools.cache``-d for hot paths in production. Tests that
    monkeypatch ``EVEROS_*`` env vars must see fresh settings on each
    function — clear both caches around every test to keep results
    deterministic regardless of declaration order.

Cross-suite fixtures:
    ``long_conversation`` lives here (not under ``tests/e2e/conftest.py``)
    because both ``tests/e2e/`` and ``tests/integration/search/`` depend
    on it — pytest conftest cascades down the directory tree, so a
    fixture defined under ``tests/e2e/`` is invisible to siblings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_LONG_CONV_PATH = _FIXTURE_DIR / "long_conversation_locomo_caroline_melanie.json"


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    import structlog

    from everos.component.utils import datetime as dt_module
    from everos.config import load_settings

    # ``configure_logging`` (called by some e2e fixtures / the CLI entry)
    # sets ``cache_logger_on_first_use=True``; once a logger is cached,
    # ``structlog.testing.capture_logs`` can no longer intercept events,
    # which silently breaks log-assertion tests that run *after* it in the
    # same process. Reset structlog to defaults around every test so that
    # global config never leaks across the suite.
    structlog.reset_defaults()
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    yield
    structlog.reset_defaults()
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()


@pytest.fixture(scope="session")
def long_conversation() -> dict:
    """LoCoMo conv_0 fixture (419 messages, 19 batches, one session)."""
    return json.loads(_LONG_CONV_PATH.read_text())
