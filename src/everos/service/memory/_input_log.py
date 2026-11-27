"""Append-only JSONL log of accepted ``/add`` requests (issue #537).

The log is deliberately independent of extraction: it records what the
client sent, before boundary detection can rewrite or drop anything. The
first PR ships it disabled; deletion/purge integration and the default-on
switch follow in a second PR.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from everos.config import InputLogSettings
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_now_with_timezone

logger = get_logger(__name__)

_LOG_DIR = ".input_log"


class InputLogWriter:
    """One process-wide writer; appends one line per accepted ``/add``."""

    def __init__(
        self,
        *,
        root: Path,
        settings: InputLogSettings,
        app_dir: Callable[[str], str] = lambda value: value,
        project_dir: Callable[[str], str] = lambda value: value,
    ) -> None:
        self._root = root
        self._settings = settings
        self._app_dir = app_dir
        self._project_dir = project_dir
        self._locks: dict[Path, asyncio.Lock] = {}

    def _lock_for(self, directory: Path) -> asyncio.Lock:
        lock = self._locks.get(directory)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[directory] = lock
        return lock

    async def log_add(self, payload: dict[str, Any], *, request_id: str) -> None:
        """Record one request; a failure is logged and never fails ``/add``."""
        if not self._settings.enabled:
            return
        try:
            await self._write(payload, request_id=request_id)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error(
                "input_log_write_failed",
                request_id=request_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _write(self, payload: dict[str, Any], *, request_id: str) -> None:
        now = get_now_with_timezone()
        directory = self._directory(payload)
        line = self._truncate(
            {
                **payload,
                "request_id": request_id,
                "received_at": now.isoformat(),
            }
        )
        async with self._lock_for(directory):
            await asyncio.to_thread(self._append_sync, directory, now, line)

    async def delete_session(
        self, *, app_id: str, project_id: str, session_id: str
    ) -> None:
        """Drop one session's lines from every daily file in the scope."""
        directory = (
            self._root
            / self._app_dir(app_id)
            / self._project_dir(project_id)
            / _LOG_DIR
        )
        async with self._lock_for(directory):
            await asyncio.to_thread(self._delete_session_sync, directory, session_id)

    def _delete_session_sync(self, directory: Path, session_id: str) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.jsonl"):
            self._rewrite_streaming(
                path,
                lambda record: record.get("session_id") != session_id,
            )

    async def delete_owner(
        self,
        *,
        app_id: str,
        project_id: str,
        owner_id: str,
        session_id: str | None = None,
    ) -> None:
        """Drop every line that contains one owner's messages.

        Privacy takes precedence over keeping other owners' raw text: a line
        with the deleted owner and another owner is removed whole. The
        ``session_id`` filter limits which lines are considered.
        """
        directory = (
            self._root
            / self._app_dir(app_id)
            / self._project_dir(project_id)
            / _LOG_DIR
        )
        async with self._lock_for(directory):
            await asyncio.to_thread(
                self._delete_owner_sync, directory, owner_id, session_id
            )

    def _delete_owner_sync(
        self, directory: Path, owner_id: str, session_id: str | None
    ) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.jsonl"):
            self._rewrite_streaming(
                path,
                lambda record: not _owner_in_line(record, owner_id, session_id),
            )

    def _rewrite_streaming(
        self, path: Path, keep: Callable[[dict[str, Any]], bool]
    ) -> None:
        """Rewrite one daily file line by line, with O(1) extra memory."""
        temp = path.with_suffix(".tmp")
        changed = False
        kept_any = False
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        if os.name == "posix":
            os.chmod(temp, 0o600)
        with (
            path.open("r", encoding="utf-8") as src,
            os.fdopen(fd, "w", encoding="utf-8") as dst,
        ):
            for line in src:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dst.write(line)
                    kept_any = True
                    continue
                if keep(record):
                    dst.write(line)
                    kept_any = True
                else:
                    changed = True
        if not changed:
            temp.unlink(missing_ok=True)
        elif kept_any:
            temp.replace(path)
        else:
            temp.unlink(missing_ok=True)
            path.unlink(missing_ok=True)

    async def purge_scope(self, scope: Any) -> None:
        """Remove every ``.input_log`` directory the purge scope covers."""
        for directory in self._purge_dirs(scope):
            async with self._lock_for(directory):
                await asyncio.to_thread(self._purge_directory_sync, directory)

    def _purge_directory_sync(self, directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            # Already gone is success; any other removal failure must escape
            # so the purge runner retries the step.
            return

    def _purge_dirs(self, scope: Any) -> list[Path]:
        if scope.kind == "space":
            return list(self._root.glob("*/*/.input_log"))
        if scope.kind == "app":
            return list((self._root / str(scope.app_dir)).glob("*/.input_log"))
        return [self._root / str(scope.app_dir) / str(scope.project_dir) / _LOG_DIR]

    def _directory(self, payload: dict[str, Any]) -> Path:
        app_id = str(payload.get("app_id", "default"))
        project_id = str(payload.get("project_id", "default"))
        return (
            self._root
            / self._app_dir(app_id)
            / self._project_dir(project_id)
            / _LOG_DIR
        )

    def _truncate(self, line: dict[str, Any]) -> dict[str, Any]:
        encoded = _encode(line)
        if len(encoded) <= self._settings.max_inline_bytes:
            return line
        for message in line.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                # The request schema also permits a plain string body.
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                value = item.get("base64")
                if not isinstance(value, str):
                    continue
                extras = dict(item.get("extras") or {})
                extras["base64_sha256"] = hashlib.sha256(
                    value.encode("utf-8")
                ).hexdigest()
                extras["base64_bytes"] = len(value)
                item["base64"] = None
                item["extras"] = extras
        return line

    def _append_sync(
        self, directory: Path, now: dt.datetime, line: dict[str, Any]
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._prune(directory, now.date())
        path = directory / f"{now.date().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_encode(line) + "\n")

    def _prune(self, directory: Path, today: dt.date) -> None:
        cutoff = today - dt.timedelta(days=self._settings.retention_days)
        for path in directory.glob("*.jsonl"):
            try:
                day = dt.date.fromisoformat(path.stem)
            except ValueError:
                continue
            if day < cutoff:
                path.unlink(missing_ok=True)


def _owner_in_line(
    record: dict[str, Any], owner_id: str, session_id: str | None
) -> bool:
    if session_id is not None and record.get("session_id") != session_id:
        return False
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and message.get("sender_id") == owner_id
        for message in messages
    )


def _encode(line: dict[str, Any]) -> str:
    return json.dumps(line, ensure_ascii=False, separators=(",", ":"))
