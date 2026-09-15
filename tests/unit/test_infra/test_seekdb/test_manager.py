"""Exercise SeekDB lifecycle, diagnostics, locking, and row normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import SeekdbSettings
from everos.infra.persistence.seekdb.errors import (
    SeekdbConfigurationError,
    SeekdbIntegrityError,
    SeekdbOperationalError,
)
from everos.infra.persistence.seekdb.seekdb_manager import (
    SeekdbSession,
    SeekdbTarget,
    resolve_target,
    table_name,
)


class OperationalError(Exception):
    """Fake DB-API operational error retaining the numeric code."""


class IntegrityError(Exception):
    """Fake DB-API integrity error."""


class _RawConnection:
    def __init__(self) -> None:
        self.pings: list[bool] = []

    def ping(self, *, reconnect: bool) -> None:
        self.pings.append(reconnect)


class _FakeServer:
    def __init__(
        self,
        rows: object = None,
        *,
        connection_error: Exception | None = None,
        execute_errors: list[Exception] | None = None,
    ) -> None:
        self.rows = rows
        self.connection_error = connection_error
        self.execute_errors = list(execute_errors or [])
        self.executed: list[str] = []
        self.raw = _RawConnection()
        self.cleanup_count = 0

    def _execute(self, sql: str) -> object:
        self.executed.append(sql)
        if self.execute_errors:
            raise self.execute_errors.pop(0)
        return self.rows

    def get_raw_connection(self) -> object:
        if self.connection_error is not None:
            raise self.connection_error
        return self.raw

    def _cleanup(self) -> None:
        self.cleanup_count += 1


def test_remote_target_requires_host_and_resolves_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SeekdbConfigurationError, match="host is required"):
        resolve_target(SeekdbSettings(mode="remote"))
    monkeypatch.setenv("SEEKDB_PASSWORD", "secret")
    target = resolve_target(SeekdbSettings(mode="remote", host="db.example"))
    assert target.host == "db.example"
    assert target.password == "secret"


def test_embedded_target_rejects_windows_and_resolves_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    monkeypatch.setattr(manager.sys, "platform", "win32")
    with pytest.raises(SeekdbConfigurationError, match="Linux and macOS"):
        resolve_target(SeekdbSettings())
    monkeypatch.setattr(manager.sys, "platform", "linux")
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    target = resolve_target(SeekdbSettings())
    assert target.path == tmp_path / ".index" / "seekdb"


def test_session_normalizes_rows_and_pings_remote_connection() -> None:
    embedded = SeekdbSession(_FakeServer([("one", 2)]), mode="embedded")
    remote_server = _FakeServer([{"ID": "one", "COUNT": 2}])
    remote = SeekdbSession(remote_server, mode="remote")
    assert embedded.fetch_all("SELECT", ["id", "count"]) == [{"id": "one", "count": 2}]
    assert remote.fetch_all("SELECT", ["id", "count"]) == [{"id": "one", "count": 2}]
    assert remote_server.raw.pings == [True]


def test_session_maps_integrity_and_preserves_operational_context() -> None:
    duplicate = SeekdbSession(
        _FakeServer(execute_errors=[IntegrityError(1062, "duplicate")]),
        mode="embedded",
    )
    with pytest.raises(SeekdbIntegrityError, match=r"INSERT.*unit_episode.*1062"):
        duplicate.execute("INSERT INTO unit_episode VALUES (1)", table="unit_episode")

    broken = SeekdbSession(
        _FakeServer(execute_errors=[OperationalError(1064, "syntax")]),
        mode="embedded",
    )
    with pytest.raises(SeekdbOperationalError, match=r"SELECT.*unit_episode.*1064"):
        broken.fetch_scalar("SELECT broken", table="unit_episode")


async def test_disconnect_invalidates_cache_and_next_get_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    failed_server = _FakeServer(
        execute_errors=[OperationalError(2013, "lost connection")]
    )
    failed = SeekdbSession(failed_server, mode="remote")
    monkeypatch.setattr(manager, "_session", failed)
    with pytest.raises(SeekdbOperationalError, match="2013"):
        failed.execute("SELECT 1", table="unit_episode")
    assert manager._session is None
    assert failed_server.cleanup_count == 1

    healthy = SeekdbSession(_FakeServer([(1,)]), mode="remote")
    monkeypatch.setattr(manager, "_open_session", lambda target: healthy)
    monkeypatch.setattr(
        manager,
        "resolve_target",
        lambda: SeekdbTarget(mode="remote", database="everos", host="db.example"),
    )
    # A coroutine may have captured the old bound method before the first
    # failure. run() rebinds it after taking the operation lock.
    assert await manager.run(failed.fetch_scalar, "SELECT 1") == 1
    assert await manager.get_session() is healthy
    await manager.dispose_connection()


def test_embedded_engine_error_does_not_trigger_remote_reconnect_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    embedded = SeekdbSession(
        _FakeServer(execute_errors=[OperationalError(2013, "engine error")]),
        mode="embedded",
    )
    monkeypatch.setattr(manager, "_session", embedded)
    with pytest.raises(SeekdbOperationalError, match="2013"):
        embedded.execute("SELECT 1")
    assert manager._session is embedded


def test_open_existing_database_never_attempts_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    server = _FakeServer()
    databases: list[str] = []

    def make_server(target: SeekdbTarget, *, database: str) -> _FakeServer:
        databases.append(database)
        return server

    monkeypatch.setattr(manager, "_new_server", make_server)
    session = manager._open_session(
        SeekdbTarget(mode="remote", database="everos", host="db.example")
    )
    assert databases == ["everos"]
    assert server.executed == [
        "SET NAMES utf8mb4 COLLATE utf8mb4_bin",
        "SET SESSION sql_mode = TRIM(BOTH ',' FROM REPLACE(CONCAT(',', "
        "@@SESSION.sql_mode, ','), ',NO_BACKSLASH_ESCAPES,', ','))",
    ]
    session.close()


def test_missing_database_is_checked_then_created_with_binary_collation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    missing = _FakeServer(connection_error=OperationalError(1049, "unknown database"))
    admin = _FakeServer(rows=[])
    connected = _FakeServer()
    servers = iter([missing, admin, connected])
    databases: list[str] = []

    def make_server(target: SeekdbTarget, *, database: str) -> _FakeServer:
        databases.append(database)
        return next(servers)

    monkeypatch.setattr(manager, "_new_server", make_server)
    session = manager._open_session(
        SeekdbTarget(mode="remote", database="everos", host="db.example")
    )
    assert databases == ["everos", "information_schema", "everos"]
    assert any(
        sql == "CREATE DATABASE IF NOT EXISTS `everos` DEFAULT CHARACTER SET "
        "utf8mb4 COLLATE utf8mb4_bin"
        for sql in admin.executed
    )
    session.close()


def test_embedded_database_creation_uses_real_default_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    missing = _FakeServer(connection_error=OperationalError(1049, "unknown database"))
    admin = _FakeServer(rows=[])
    connected = _FakeServer()
    servers = iter([missing, admin, connected])
    databases: list[str] = []

    def make_server(target: SeekdbTarget, *, database: str) -> _FakeServer:
        databases.append(database)
        return next(servers)

    monkeypatch.setattr(manager, "_new_server", make_server)
    server = manager._connect_target_database(
        SeekdbTarget(mode="embedded", database="everos", path=tmp_path)
    )
    assert databases == ["everos", "test", "everos"]
    server._cleanup()


def test_missing_database_permission_error_has_actionable_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    missing = _FakeServer(connection_error=OperationalError(1049, "unknown database"))

    class CreateDeniedServer(_FakeServer):
        def _execute(self, sql: str) -> object:
            self.executed.append(sql)
            if sql.startswith("CREATE DATABASE"):
                raise OperationalError(1044, "denied")
            return []

    admin = CreateDeniedServer()
    servers = iter([missing, admin])
    monkeypatch.setattr(
        manager,
        "_new_server",
        lambda target, *, database: next(servers),
    )
    with pytest.raises(SeekdbConfigurationError, match=r"pre-create.*grant CREATE"):
        manager._open_session(
            SeekdbTarget(mode="remote", database="everos", host="db.example")
        )


def test_embedded_directory_lock_fails_before_engine_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    class BusyFcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        @staticmethod
        def flock(fd: int, operation: int) -> None:
            raise BlockingIOError

    monkeypatch.setattr(manager.importlib, "import_module", lambda name: BusyFcntl)
    with pytest.raises(SeekdbConfigurationError, match="already opened"):
        manager._acquire_embedded_lock(
            SeekdbTarget(mode="embedded", database="everos", path=tmp_path)
        )


def test_embedded_directory_lock_is_released_on_close_and_open_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import everos.infra.persistence.seekdb.seekdb_manager as manager

    class FakeLock:
        def __init__(self) -> None:
            self.releases = 0

        def release(self) -> None:
            self.releases += 1

    target = SeekdbTarget(mode="embedded", database="everos", path=tmp_path)
    held = FakeLock()
    server = _FakeServer()
    monkeypatch.setattr(manager, "_acquire_embedded_lock", lambda target: held)
    monkeypatch.setattr(manager, "_connect_target_database", lambda target: server)
    session = manager._open_session(target)
    session.close()
    assert held.releases == 1

    failed = FakeLock()
    monkeypatch.setattr(manager, "_acquire_embedded_lock", lambda target: failed)

    def fail_to_connect(target: SeekdbTarget) -> _FakeServer:
        raise RuntimeError("engine failed")

    monkeypatch.setattr(manager, "_connect_target_database", fail_to_connect)
    with pytest.raises(RuntimeError, match="engine failed"):
        manager._open_session(target)
    assert failed.releases == 1


def test_table_name_validates_the_configured_prefix() -> None:
    assert table_name("episode", SeekdbSettings(table_prefix="tenant_a")) == (
        "tenant_a_episode"
    )
