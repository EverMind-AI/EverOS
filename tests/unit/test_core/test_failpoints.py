"""The four failpoint modes: crash, raise n times, pause until released, skip."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy.exc

from everos.core import failpoints
from everos.core.errors import VectorStoreBusyError


@pytest.fixture
def arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Callable[[str], ModuleType]]:
    """``arm("a=pause,b")``: the module re-read with that environment."""

    def _arm(spec: str) -> ModuleType:
        monkeypatch.setenv("EVEROS_TEST_FAILPOINTS", spec)
        monkeypatch.setenv("EVEROS_TEST_FAILPOINT_DIR", str(tmp_path))
        return importlib.reload(failpoints)

    yield _arm
    monkeypatch.delenv("EVEROS_TEST_FAILPOINTS", raising=False)
    importlib.reload(failpoints)


async def test_unarmed_is_noop(arm: Callable[[str], ModuleType]) -> None:
    fp = arm("")
    fp.failpoint("x")
    assert await fp.afailpoint("x") is False


async def test_raise_alias_fires_n_times(arm: Callable[[str], ModuleType]) -> None:
    fp = arm("x=raise:busy:2")
    for _ in range(2):
        with pytest.raises(VectorStoreBusyError, match="failpoint x"):
            await fp.afailpoint("x")
    assert await fp.afailpoint("x") is False


def test_sqlite_locked_alias_message(arm: Callable[[str], ModuleType]) -> None:
    fp = arm("x=raise:sqlite_locked:1")
    with pytest.raises(sqlalchemy.exc.OperationalError) as caught:
        fp.failpoint("x")
    assert "database is locked" in str(caught.value)


async def test_pause_waits_for_release_file(
    arm: Callable[[str], ModuleType], tmp_path: Path
) -> None:
    fp = arm("x=pause")
    waiter = asyncio.create_task(fp.afailpoint("x"))
    await asyncio.sleep(0.3)
    assert not waiter.done()
    (tmp_path / "x.release").write_text("go")
    assert await asyncio.wait_for(waiter, timeout=2) is False


async def test_skip_returns_true(arm: Callable[[str], ModuleType]) -> None:
    fp = arm("x=skip,y")
    assert await fp.afailpoint("x") is True
    fp.failpoint("x")  # a synchronous point cannot skip: nothing happens


def test_bare_name_still_kills_with_137() -> None:
    code = subprocess.run(
        [
            sys.executable,
            "-c",
            "from everos.core.failpoints import failpoint; failpoint('x')",
        ],
        env={"EVEROS_TEST_FAILPOINTS": "x", "PATH": ""},
        check=False,
    ).returncode
    assert code == 137
