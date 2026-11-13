"""A lance query-execution failure that a retry clears must reach the cascade
worker as :class:`VectorStoreBusyError` (retried with backoff), not as a bare
``RuntimeError`` (filed as unrecoverable, needing a manual ``cascade fix``).

A full disk is transient too: once space is freed the same write lands. In
either form -- lance's ``RuntimeError`` naming it, or an ``OSError`` with
``ENOSPC`` -- it reaches the worker as :class:`IndexDiskFullError`, which the
worker waits out with a back-off instead of filing the row as permanent.
"""

from __future__ import annotations

import errno
from typing import ClassVar

import pytest

from everos.core.errors import IndexDiskFullError, VectorStoreBusyError
from everos.infra.persistence.lancedb import LanceRepoBase
from everos.infra.persistence.lancedb._base import repository as repo_mod
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


_FULL = (
    "LanceError(IO): Generic LocalFileSystem error: Unable to write: "
    "No space left on device (os error 28)"
)


def test_only_the_spill_phrase_counts_as_an_execution_error() -> None:
    assert repo_mod._is_transient_execution_error(RuntimeError(_SPILL))
    assert not repo_mod._is_transient_execution_error(RuntimeError(_FULL))


def test_a_full_disk_is_named_in_either_form() -> None:
    assert repo_mod._is_disk_full(RuntimeError(_FULL))
    assert repo_mod._is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    assert not repo_mod._is_disk_full(OSError(errno.EACCES, "Permission denied"))
    assert not repo_mod._is_disk_full(RuntimeError("LanceError(IO): file not found"))


async def test_spill_failure_inside_a_read_becomes_a_busy_error() -> None:
    with pytest.raises(VectorStoreBusyError, match="transient lance execution"):
        async with _repo()._deadline(1.0, "find_where"):
            raise RuntimeError(_SPILL)


async def test_spill_failure_inside_a_write_becomes_a_busy_error() -> None:
    """The soak's spill failures came out of merge_insert (the write path)."""
    with pytest.raises(VectorStoreBusyError, match="transient lance execution"):
        async with _repo()._locked(1.0, "upsert"):
            raise RuntimeError(_SPILL)


@pytest.mark.parametrize(
    "error",
    [RuntimeError(_FULL), OSError(errno.ENOSPC, "No space left on device")],
    ids=["lance-runtime-error", "oserror"],
)
async def test_a_full_disk_inside_a_write_is_a_disk_full_error(
    error: BaseException,
) -> None:
    with pytest.raises(IndexDiskFullError, match="the disk is full") as raised:
        async with _repo()._locked(1.0, "upsert"):
            raise error
    assert isinstance(raised.value, VectorStoreBusyError)  # transient, retried
    assert raised.value.__cause__ is error


async def test_a_full_disk_inside_a_read_is_a_disk_full_error() -> None:
    with pytest.raises(IndexDiskFullError):
        async with _repo()._deadline(1.0, "find_where"):
            raise OSError(errno.ENOSPC, "No space left on device")


async def test_other_errors_still_propagate_unchanged() -> None:
    with pytest.raises(RuntimeError, match="file not found"):
        async with _repo()._deadline(1.0, "find_where"):
            raise RuntimeError("lance error: LanceError(IO): file not found")
    with pytest.raises(PermissionError):
        async with _repo()._locked(1.0, "upsert"):
            raise PermissionError(errno.EACCES, "Permission denied")
