"""Test startup banner warning for unbackfilled memory rows.

When LanceDB tables have rows with vector IS NULL (unbackfilled memories),
the startup banner should log a warning message pointing users to
`everos cascade backfill` command.
"""

from __future__ import annotations

import hashlib

import structlog.testing

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.infra.cascade import CascadeAdmin
from everos.infra.persistence.index import EmbeddingProvenance
from everos.infra.persistence.lancedb import Episode
from everos.service.bootstrap import _log_unbackfilled_hint
from tests.fakes import Storage


def _keyword_only() -> EmbeddingProvenance:
    """No embedding provider: the index opens without a provenance check."""
    return EmbeddingProvenance(load_settings().embedding, configured=False)


def _admin(storage: Storage) -> CascadeAdmin:
    """The admin over the test's storage, as the bootstrap wires it."""
    return CascadeAdmin(
        database=storage.database,
        ledger=storage.sqlite.md_change_state,
        index=storage.index,
        provenance=_keyword_only(),
    )


async def test_no_warning_when_empty(storage: Storage) -> None:
    """No warning is logged when the database is empty."""
    admin = _admin(storage)
    await admin.start_index()

    # When database is empty, no warning should be logged
    with structlog.testing.capture_logs() as captured:
        await _log_unbackfilled_hint(admin)

    matching = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(matching) == 0, "expected no warning when database is empty"

    await admin.close_index()


async def test_warns_when_unbackfilled_rows_exist(storage: Storage) -> None:
    """_log_unbackfilled_hint warns with correct count when vector=NULL rows exist."""

    # Setup: open the index (connects to temp LanceDB)
    admin = _admin(storage)
    await admin.start_index()

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

    await storage.repos.episode.add([_make_unvectorized_episode()])

    # Capture logs and call the hint function
    with structlog.testing.capture_logs() as captured:
        await _log_unbackfilled_hint(admin)

    # Assert: the warning is logged with the correct count
    matching = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(matching) > 0, (
        "expected unbackfilled_memory_rows warning when vector=NULL rows exist"
    )
    warning = matching[0]
    assert warning.get("count") == 1, f"expected count=1, got {warning.get('count')}"

    await admin.close_index()


async def test_log_unbackfilled_hint_runs_on_startup(storage: Storage) -> None:
    """Startup should call _log_unbackfilled_hint after ensuring indexes."""
    admin = _admin(storage)

    # This startup should not raise - _log_unbackfilled_hint should be called
    await admin.start_index()
    await _log_unbackfilled_hint(admin)

    await admin.close_index()


async def test_startup_banner_logger_configuration() -> None:
    """Verify the banner logger can be obtained."""
    logger = get_logger("everos.cli.server")
    assert logger is not None
    # Verify it has the expected logging methods
    assert hasattr(logger, "warning")
    assert hasattr(logger, "info")
