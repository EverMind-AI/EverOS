"""A lance query-execution failure that a retry clears must reach the cascade
worker as :class:`VectorStoreBusyError` (retried with backoff), not as a bare
``RuntimeError`` (filed as unrecoverable, needing a manual ``cascade fix``).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from everos.core.errors import VectorStoreBusyError
from everos.infra.persistence.lancedb import LanceRepoBase
from everos.infra.persistence.lancedb.base import repository as repo_mod
from everos.infra.persistence.records import IndexRecord
from tests.fakes import TEST_DIM, UnusedPort

_SPILL = (
    "lance error: LanceError(IO): Execution error: Spill has sent an error, "
    "C:\\Users\\x\\everos-src\\.venv\\Lib\\site-packages\\lance\\..."
)


class _Note(IndexRecord):
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
    TABLE_NAME: ClassVar[str] = "_note"
    id: str


class _Repo(LanceRepoBase[_Note]):
    schema = _Note


def _repo() -> _Repo:
    """A repo whose table is never resolved: these tests stay in the guards."""
    return _Repo(source=UnusedPort("table source"), dim=TEST_DIM)


def test_only_the_spill_phrase_counts_as_transient() -> None:
    assert repo_mod._is_transient_execution_error(RuntimeError(_SPILL))
    assert not repo_mod._is_transient_execution_error(
        RuntimeError("lance error: LanceError(IO): No space left on device")
    )


async def test_spill_failure_inside_a_read_becomes_a_busy_error() -> None:
    with pytest.raises(VectorStoreBusyError, match="transient lance execution"):
        async with _repo()._deadline(1.0, "find_where"):
            raise RuntimeError(_SPILL)


async def test_spill_failure_inside_a_write_becomes_a_busy_error() -> None:
    """The soak's spill failures came out of merge_insert (the write path)."""
    with pytest.raises(VectorStoreBusyError, match="transient lance execution"):
        async with _repo()._locked(1.0, "upsert"):
            raise RuntimeError(_SPILL)


async def test_other_runtime_errors_still_propagate_unchanged() -> None:
    with pytest.raises(RuntimeError, match="No space left"):
        async with _repo()._deadline(1.0, "find_where"):
            raise RuntimeError("lance error: LanceError(IO): No space left on device")
