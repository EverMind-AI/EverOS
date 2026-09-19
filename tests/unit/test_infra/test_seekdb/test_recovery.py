"""Regression coverage for cancellation, physical reconnects and log privacy."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from everos.infra.persistence.seekdb import seekdb_manager as manager
from everos.infra.persistence.seekdb.errors import SeekdbOperationalError

from .test_manager import OperationalError, _FakeServer, _RawConnection


@pytest.fixture(autouse=True)
def _isolated_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "_session", None)
    monkeypatch.setattr(manager, "_operation_lock", asyncio.Lock())
    monkeypatch.setattr(manager, "_connection_lock", asyncio.Lock())
    monkeypatch.setattr(
        manager,
        "resolve_target",
        lambda: manager.SeekdbTarget(mode="remote", database="unit", host="db.example"),
    )


@pytest.mark.parametrize("next_operation", ["query", "shutdown"])
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_sql_finishes_before_next_operation(
    monkeypatch: pytest.MonkeyPatch, next_operation: str, worker_fails: bool
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    observed: list[bool] = []

    class Server(_FakeServer):
        def _execute(self, sql: str) -> object:
            if sql == "slow":
                started.set()
                assert release.wait(5)
                finished.set()
                if worker_fails:
                    raise OperationalError(1064, "failed after cancellation")
            else:
                observed.append(finished.is_set())
            return []

        def _cleanup(self) -> None:
            observed.append(finished.is_set())

    session = manager.SeekdbSession(Server(), mode="embedded")
    monkeypatch.setattr(manager, "_session", session)
    first = asyncio.create_task(manager.run(session.execute, "slow"))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()  # Shutdown may cancel an already cancelling request.
        second = asyncio.create_task(
            manager.dispose_connection()
            if next_operation == "shutdown"
            else manager.run(session.execute, "next")
        )
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert observed == []
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        if second is not None:
            await second
    assert observed == [True]


async def test_cancelled_open_keeps_session_owned_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    server = _FakeServer()
    session = manager.SeekdbSession(server, mode="embedded")

    def open_session(target: manager.SeekdbTarget) -> manager.SeekdbSession:
        started.set()
        assert release.wait(5)
        return session

    monkeypatch.setattr(manager, "_open_session", open_session)
    task = asyncio.create_task(manager.get_session())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await manager.get_session() is session
    await manager.dispose_connection()
    assert server.cleanup_count == 1


async def test_cancelled_close_holds_connection_lock_until_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class Server(_FakeServer):
        def _cleanup(self) -> None:
            started.set()
            assert release.wait(5)
            super()._cleanup()

    server = Server()
    monkeypatch.setattr(
        manager, "_session", manager.SeekdbSession(server, mode="embedded")
    )
    task = asyncio.create_task(manager.dispose_connection())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert manager._connection_lock.locked()
        assert manager._operation_lock.locked()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert server.cleanup_count == 1
    assert manager._session is None


async def test_failed_ping_reopens_configured_session_without_replaying_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadConnection(_RawConnection):
        def ping(self, *, reconnect: bool) -> None:
            assert reconnect is False
            raise OperationalError(2006, "gone away")

    old = _FakeServer()
    old.raw = DeadConnection()
    failed = manager.SeekdbSession(old, mode="remote")
    monkeypatch.setattr(manager, "_session", failed)
    with pytest.raises(SeekdbOperationalError, match="2006"):
        await manager.run(failed.execute, "INSERT first")
    assert old.executed == []
    assert old.cleanup_count == 1
    assert manager._session is None

    new = _FakeServer()
    monkeypatch.setattr(manager, "_new_server", lambda target, *, database: new)
    await manager.run(failed.execute, "INSERT second")
    assert new.executed[0].startswith("SET NAMES")
    assert new.executed[1].startswith("SET SESSION sql_mode")
    assert new.executed[2:] == ["INSERT second"]
    await manager.dispose_connection()


def test_driver_replacement_is_configured_before_sql() -> None:
    class Connection(_RawConnection):
        no_backslash_escapes = True

    class Server(_FakeServer):
        def _execute(self, sql: str) -> object:
            if sql.startswith("SET SESSION sql_mode"):
                self.raw.no_backslash_escapes = False
            if sql.startswith("INSERT"):
                assert self.raw.no_backslash_escapes is False
            return super()._execute(sql)

    server = Server()
    session = manager.SeekdbSession(server, mode="remote", database="unit")
    server.raw = Connection()  # pyseekdb replaced a closed physical connection.
    session.execute("INSERT first")
    session.execute("INSERT second")
    assert len(server.executed) == 4
    assert server.executed[0].startswith("SET NAMES")
    assert server.executed[1].startswith("SET SESSION sql_mode")
    assert server.executed[2:] == ["INSERT first", "INSERT second"]
    assert server.raw.pings == [False, False]


@pytest.mark.parametrize(
    "stage",
    [
        "sql",
        "integrity",
        "connect",
        "configure",
        "create",
        "admin_connect",
        "close",
        "configure_close",
    ],
)
def test_outer_exception_log_does_not_expose_driver_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, stage: str
) -> None:
    secret = "PRIVATE_MEMORY_SENTINEL"
    target = manager.SeekdbTarget(mode="remote", database="unit", host="db.example")
    error = OperationalError(1062 if stage == "integrity" else 1064, secret)
    server = _FakeServer(execute_errors=[error])
    monkeypatch.setattr(manager, "_new_server", lambda target, *, database: server)
    if stage == "connect":
        server.connection_error = error
    elif stage in {"create", "admin_connect"}:
        if stage == "admin_connect":
            server.connection_error = error
        missing = _FakeServer(connection_error=OperationalError(1049, "missing"))
        servers = iter([missing, server])
        monkeypatch.setattr(
            manager, "_new_server", lambda target, *, database: next(servers)
        )
    elif stage in {"close", "configure_close"}:

        def fail_cleanup() -> None:
            raise error

        monkeypatch.setattr(server, "_cleanup", fail_cleanup)

    with caplog.at_level(logging.ERROR):
        try:
            if stage in {
                "connect",
                "configure",
                "create",
                "admin_connect",
                "configure_close",
            }:
                manager._open_session(target)
            elif stage == "close":
                manager.SeekdbSession(server, mode="embedded").close()
            else:
                manager.SeekdbSession(server, mode="embedded").execute("INSERT test")
        except Exception:
            logging.getLogger("seekdb-review-caller").exception("operation failed")
        else:
            pytest.fail("expected an adapter exception")
    assert "operation failed" in caplog.text
    assert "Seekdb" in caplog.text
    assert secret not in caplog.text
