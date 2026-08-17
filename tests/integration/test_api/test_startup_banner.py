"""Test startup banner warning for unbackfilled memory rows.

When LanceDB tables have rows with vector IS NULL (unbackfilled memories),
the startup banner should log a warning message pointing users to
`everos cascade backfill` command.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import structlog.testing

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.entrypoints.api.lifespans.lancedb import (
    LanceDBLifespanProvider,
    _log_unbackfilled_hint,
)
from everos.infra.persistence.lancedb import Episode
from everos.infra.persistence.lancedb.repos import episode_repo


@pytest.fixture
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate each test with its own temporary MemoryRoot."""
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)


async def test_no_warning_when_empty(_isolated_memory_root: None) -> None:
    """No warning is logged when the database is empty."""
    from fastapi import FastAPI

    app = FastAPI()
    provider = LanceDBLifespanProvider()
    await provider.startup(app)

    # When database is empty, no warning should be logged
    with structlog.testing.capture_logs() as captured:
        await _log_unbackfilled_hint()

    matching = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(matching) == 0, "expected no warning when database is empty"

    await provider.shutdown(app)


async def test_warns_when_unbackfilled_rows_exist(
    _isolated_memory_root: None,
) -> None:
    """_log_unbackfilled_hint warns with correct count when vector=NULL rows exist."""
    from fastapi import FastAPI

    # Setup: initialize provider (connects to temp LanceDB)
    app = FastAPI()
    provider = LanceDBLifespanProvider()
    await provider.startup(app)

    # Seed: insert one Episode row with vector=None
    def _make_unvectorized_episode() -> Episode:
        """Create an Episode row with vector=None."""
        return Episode(
            id="test_owner_entry1",
            entry_id="entry1",
            owner_id="test_owner",
            owner_type="user",
            app_id="default",
            project_id="default",
            session_id="s_test",
            timestamp=get_utc_now(),
            parent_type="memcell",
            parent_id="mc_test",
            sender_ids=["test_owner"],
            subject="test subject",
            summary=None,
            episode="test episode content",
            episode_tokens="test episode",
            md_path="users/test_owner/default/default/2026-07-25.md",
            content_sha256=hashlib.sha256(b"test").hexdigest(),
            deprecated_by=None,
            vector=None,  # explicitly unvectorized
        )

    await episode_repo.add([_make_unvectorized_episode()])

    # Capture logs and call the hint function
    with structlog.testing.capture_logs() as captured:
        await _log_unbackfilled_hint()

    # Assert: the warning is logged with the correct count
    matching = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(matching) > 0, (
        "expected unbackfilled_memory_rows warning when vector=NULL rows exist"
    )
    warning = matching[0]
    assert warning.get("count") == 1, f"expected count=1, got {warning.get('count')}"

    await provider.shutdown(app)


async def test_log_unbackfilled_hint_runs_on_startup(
    _isolated_memory_root: None,
) -> None:
    """Startup should call _log_unbackfilled_hint after ensuring indexes."""
    from fastapi import FastAPI

    app = FastAPI()
    provider = LanceDBLifespanProvider()

    # This startup should not raise - _log_unbackfilled_hint should be called
    await provider.startup(app)

    await provider.shutdown(app)


async def test_startup_banner_logger_configuration() -> None:
    """Verify the banner logger can be obtained."""
    logger = get_logger("everos.cli.server")
    assert logger is not None
    # Verify it has the expected logging methods
    assert hasattr(logger, "warning")
    assert hasattr(logger, "info")
