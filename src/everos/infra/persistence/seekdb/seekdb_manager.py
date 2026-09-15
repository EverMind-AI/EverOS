"""Own the optional pyseekdb connection and serialize SQL execution.

Both the embedded DB-API connection and PyMySQL connections are treated as
single-threaded resources. The async boundary therefore holds one process-wide
lock while offloading blocking work, and only this module imports pyseekdb.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from everos.config import SeekdbSettings, load_settings, resolve_root
from everos.core.observability.logging import get_logger

from .errors import (
    SeekdbConfigurationError,
    SeekdbIntegrityError,
    SeekdbOperationalError,
)
from .sql import literal, quote_identifier

logger = get_logger(__name__)


class _SqlServer(Protocol):
    """The pyseekdb 1.4.x private surface isolated by this adapter."""

    def _execute(self, sql: str) -> Any: ...

    def get_raw_connection(self) -> Any: ...

    def _cleanup(self) -> None: ...


class _HeldFileLock(Protocol):
    """Lifetime lock retained by an embedded session."""

    def release(self) -> None: ...


@dataclass(frozen=True)
class SeekdbTarget:
    """A fully resolved embedded directory or remote endpoint."""

    mode: Literal["embedded", "remote"]
    database: str
    path: Path | None = None
    host: str | None = None
    port: int = 2881
    tenant: str = ""
    user: str = "root"
    password: str = ""
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0


class SeekdbSession:
    """Normalize pyseekdb SQL result rows without relying on cursor metadata."""

    def __init__(
        self,
        server: _SqlServer,
        *,
        mode: Literal["embedded", "remote"],
        directory_lock: _HeldFileLock | None = None,
    ):
        self._server = server
        self._directory_lock = directory_lock
        self._invalidated = False
        self.mode = mode

    def execute(self, sql: str, *, table: str | None = None) -> None:
        self._call(sql, table=table)

    def fetch_all(
        self,
        sql: str,
        columns: Sequence[str],
        *,
        table: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._call(sql, table=table) or []
        return [_normalize_row(row, columns) for row in rows]

    def fetch_scalar(self, sql: str, *, table: str | None = None) -> Any:
        rows = self._call(sql, table=table) or []
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, Mapping):
            return next(iter(row.values()), None)
        if isinstance(row, (tuple, list)):
            return row[0] if row else None
        return row

    def close(self) -> None:
        try:
            self._server._cleanup()
        finally:
            if self._directory_lock is not None:
                self._directory_lock.release()
                self._directory_lock = None

    def _call(self, sql: str, *, table: str | None) -> Any:
        try:
            if self.mode == "remote":
                raw = self._server.get_raw_connection()
                ping = getattr(raw, "ping", None)
                if callable(ping):
                    ping(reconnect=True)
            return self._server._execute(sql)
        except Exception as exc:
            kind = _sql_kind(sql)
            code = _error_code(exc)
            disconnected = self.mode == "remote" and code in {2006, 2013}
            if disconnected:
                self._invalidate()
            logger.warning(
                "seekdb_operation_failed",
                sql_kind=kind,
                table=table,
                error_type=type(exc).__name__,
                error_code=code,
                session_invalidated=disconnected,
            )
            detail = _error_detail(exc, code)
            if _is_integrity_error(exc):
                raise SeekdbIntegrityError(
                    f"SeekDB rejected a {kind} operation on {table or 'SeekDB'} "
                    f"due to an integrity constraint [{detail}]"
                ) from exc
            raise SeekdbOperationalError(
                f"SeekDB failed to execute a {kind} operation on "
                f"{table or 'SeekDB'} [{detail}]"
            ) from exc

    def _invalidate(self) -> None:
        global _session
        if self._invalidated:
            return
        self._invalidated = True
        if _session is self:
            _session = None
        try:
            self._server._cleanup()
        except Exception:
            logger.warning("seekdb_disconnected_session_cleanup_failed")


@dataclass
class _EmbeddedDirectoryLock:
    """A non-blocking POSIX flock held for an embedded engine's lifetime."""

    fd: int
    path: Path
    fcntl: Any

    def release(self) -> None:
        if self.fd < 0:
            return
        fd, self.fd = self.fd, -1
        try:
            self.fcntl.flock(fd, self.fcntl.LOCK_UN)
        finally:
            os.close(fd)


_session: SeekdbSession | None = None
_connection_lock = asyncio.Lock()
_operation_lock = asyncio.Lock()


def resolve_target(settings: SeekdbSettings | None = None) -> SeekdbTarget:
    """Validate settings and resolve the default embedded data directory."""
    cfg = settings or load_settings().seekdb
    password = cfg.password.get_secret_value() or os.environ.get("SEEKDB_PASSWORD", "")
    if cfg.mode == "embedded":
        if sys.platform not in {"linux", "darwin"}:
            raise SeekdbConfigurationError(
                "Embedded SeekDB requires a pylibseekdb wheel, which is currently "
                "available on Linux and macOS only. Use remote mode, Docker, or WSL2."
            )
        path = Path(cfg.path).expanduser() if cfg.path else _default_path()
        return SeekdbTarget(mode="embedded", database=cfg.database, path=path)
    if not cfg.host.strip():
        raise SeekdbConfigurationError("[seekdb] host is required when mode = 'remote'")
    return SeekdbTarget(
        mode="remote",
        database=cfg.database,
        host=cfg.host.strip(),
        port=cfg.port,
        tenant=cfg.tenant,
        user=cfg.user,
        password=password,
        connect_timeout_seconds=cfg.connect_timeout_seconds,
        read_timeout_seconds=cfg.read_timeout_seconds,
    )


def table_name(logical: str, settings: SeekdbSettings | None = None) -> str:
    """Return the configured physical table name after identifier validation."""
    cfg = settings or load_settings().seekdb
    name = f"{cfg.table_prefix}_{logical}"
    quote_identifier(name)
    return name


async def get_session() -> SeekdbSession:
    """Create and cache the configured SeekDB session."""
    global _session
    if _session is not None:
        return _session
    async with _connection_lock:
        if _session is None:
            target = resolve_target()
            _session = await asyncio.to_thread(_open_session, target)
            logger.info(
                "seekdb_connection_opened",
                mode=target.mode,
                database=target.database,
                host=target.host,
            )
        return _session


async def run[**P, R](fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """Run one blocking client operation under the connection lock."""
    async with _operation_lock:
        owner = getattr(fn, "__self__", None)
        if isinstance(owner, SeekdbSession) and owner._invalidated:
            current = await get_session()
            rebound = cast(Callable[P, R], getattr(current, fn.__name__))
            return await asyncio.to_thread(rebound, *args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)


async def dispose_connection() -> None:
    """Close the process session and permit a fresh target on next use."""
    global _session
    async with _connection_lock:
        session = _session
        _session = None
    if session is not None:
        async with _operation_lock:
            await asyncio.to_thread(session.close)
        logger.info("seekdb_connection_closed")


def _open_session(target: SeekdbTarget) -> SeekdbSession:
    directory_lock = _acquire_embedded_lock(target)
    try:
        server = _connect_target_database(target)
        _configure_session(server, target)
        return SeekdbSession(
            server,
            mode=target.mode,
            directory_lock=directory_lock,
        )
    except BaseException:
        if directory_lock is not None:
            directory_lock.release()
        raise


def _connect_target_database(target: SeekdbTarget) -> _SqlServer:
    try:
        return _new_connected_server(target, database=target.database)
    except SeekdbConfigurationError:
        raise
    except Exception as exc:
        if not _is_missing_database_error(exc):
            raise SeekdbOperationalError(
                f"Could not connect to SeekDB database {target.database!r} "
                f"[{_error_detail(exc, _error_code(exc))}]"
            ) from exc
    _ensure_database(target)
    try:
        return _new_connected_server(target, database=target.database)
    except Exception as exc:
        raise SeekdbOperationalError(
            f"Could not connect to newly created SeekDB database "
            f"{target.database!r} [{_error_detail(exc, _error_code(exc))}]"
        ) from exc


def _new_connected_server(target: SeekdbTarget, *, database: str) -> _SqlServer:
    try:
        server = _new_server(target, database=database)
    except RuntimeError as exc:
        if target.mode == "embedded" and _is_driver_unavailable(exc):
            raise SeekdbConfigurationError(
                "SeekDB embedded support is unavailable. Install "
                "everos[seekdb-embedded]."
            ) from exc
        raise
    try:
        server.get_raw_connection()
        return server
    except Exception:
        server._cleanup()
        raise


def _ensure_database(target: SeekdbTarget) -> None:
    # Embedded seekdb guarantees its default ``test`` database; selecting a
    # virtual system schema as the initial embedded database is not part of
    # pylibseekdb's public contract. Remote MySQL endpoints can select
    # information_schema directly without requiring access to another user DB.
    admin_database = "test" if target.mode == "embedded" else "information_schema"
    server = _new_connected_server(target, database=admin_database)
    try:
        rows = server._execute(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
            f"WHERE SCHEMA_NAME = {literal(target.database)}"
        )
        if rows:
            return
        server._execute(
            f"CREATE DATABASE IF NOT EXISTS {quote_identifier(target.database)} "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
        )
    except Exception as exc:
        if _is_permission_error(exc):
            raise SeekdbConfigurationError(
                f"SeekDB database {target.database!r} does not exist and "
                "the configured account cannot create it; pre-create the "
                "database with utf8mb4_bin collation or grant CREATE permission"
            ) from exc
        raise SeekdbOperationalError(
            f"Could not ensure SeekDB database {target.database!r} "
            f"[{_error_detail(exc, _error_code(exc))}]"
        ) from exc
    finally:
        server._cleanup()


def _configure_session(server: _SqlServer, target: SeekdbTarget) -> None:
    try:
        server._execute("SET NAMES utf8mb4 COLLATE utf8mb4_bin")
        server._execute(
            "SET SESSION sql_mode = TRIM(BOTH ',' FROM REPLACE(CONCAT(',', "
            "@@SESSION.sql_mode, ','), ',NO_BACKSLASH_ESCAPES,', ','))"
        )
    except Exception as exc:
        server._cleanup()
        raise SeekdbConfigurationError(
            f"Could not configure the SeekDB session for {target.database!r}; "
            "the account must be allowed to set its session charset and sql_mode"
        ) from exc


def _acquire_embedded_lock(target: SeekdbTarget) -> _EmbeddedDirectoryLock | None:
    if target.mode != "embedded":
        return None
    assert target.path is not None
    target.path.mkdir(parents=True, exist_ok=True)
    lock_path = target.path / ".everos.lock"
    fcntl = importlib.import_module("fcntl")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise SeekdbConfigurationError(
            f"embedded SeekDB at {target.path} is already opened by another "
            "process; use remote mode or stop the other process"
        ) from exc
    except BaseException:
        os.close(fd)
        raise
    return _EmbeddedDirectoryLock(fd, lock_path, fcntl)


def _new_server(target: SeekdbTarget, *, database: str) -> _SqlServer:
    try:
        client = importlib.import_module("pyseekdb.client")
        if target.mode == "embedded":
            cls = client.SeekdbEmbeddedClient
            assert target.path is not None
            return cls(path=str(target.path), database=database)
        cls = client.RemoteServerClient
        assert target.host is not None
        options: dict[str, Any] = dict(
            host=target.host,
            port=target.port,
            database=database,
            user=target.user,
            password=target.password,
            connect_timeout=target.connect_timeout_seconds,
            read_timeout=target.read_timeout_seconds,
            write_timeout=target.read_timeout_seconds,
        )
        if target.tenant:
            options["tenant"] = target.tenant
        return cls(**options)
    except (ImportError, AttributeError) as exc:
        extra = "seekdb-embedded" if target.mode == "embedded" else "seekdb"
        raise SeekdbConfigurationError(
            f"SeekDB {target.mode} support is unavailable. Install everos[{extra}]."
        ) from exc


def _default_path() -> Path:
    return resolve_root() / ".index" / "seekdb"


def _normalize_row(row: Any, columns: Sequence[str]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        folded = {str(key).casefold(): value for key, value in row.items()}
        return {name: folded.get(name.casefold()) for name in columns}
    if isinstance(row, (tuple, list)):
        if len(row) != len(columns):
            raise SeekdbOperationalError(
                f"SeekDB returned {len(row)} values for {len(columns)} columns"
            )
        return dict(zip(columns, row, strict=True))
    if len(columns) == 1:
        return {columns[0]: row}
    raise SeekdbOperationalError(
        f"SeekDB returned an unsupported row type: {type(row).__name__}"
    )


def _sql_kind(sql: str) -> str:
    return sql.lstrip().split(maxsplit=1)[0].upper() if sql.strip() else "SQL"


def _is_integrity_error(exc: Exception) -> bool:
    if "integrity" in type(exc).__name__.casefold():
        return True
    return _error_code(exc) in {1062, 1586}


def _is_missing_database_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return _error_code(exc) == 1049 or any(
        marker in message for marker in ("unknown database", "database does not exist")
    )


def _is_permission_error(exc: Exception) -> bool:
    return _error_code(exc) in {1044, 1045, 1142, 1227} or any(
        marker in str(exc).casefold()
        for marker in ("access denied", "permission denied")
    )


def _is_driver_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "pylibseekdb" in message and any(
        marker in message for marker in ("not found", "not installed", "unavailable")
    )


def _error_code(exc: Exception) -> int | None:
    for name in ("errno", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return abs(value)
    for value in exc.args:
        if isinstance(value, int) and not isinstance(value, bool):
            return abs(value)
    match = re.search(r"(?:error|code)\s*[=:]?\s*-?(\d+)", str(exc), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _error_detail(exc: Exception, code: int | None) -> str:
    name = type(exc).__name__
    return f"{name} {code}" if code is not None else name


__all__ = [
    "SeekdbSession",
    "SeekdbTarget",
    "dispose_connection",
    "get_session",
    "resolve_target",
    "run",
    "table_name",
]
