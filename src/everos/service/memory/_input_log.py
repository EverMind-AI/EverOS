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
        self._lock = asyncio.Lock()

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
        async with self._lock:
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
        async with self._lock:
            await asyncio.to_thread(self._delete_session_sync, directory, session_id)

    def _delete_session_sync(self, directory: Path, session_id: str) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.jsonl"):
            original = path.read_text(encoding="utf-8").splitlines()
            kept: list[str] = []
            for line in original:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if record.get("session_id") != session_id:
                    kept.append(line)
            if not kept:
                path.unlink(missing_ok=True)
            elif len(kept) != len(original):
                temp = path.with_suffix(".tmp")
                temp.write_text("\n".join(kept) + "\n", encoding="utf-8")
                temp.replace(path)

    async def purge_scope(self, scope: Any) -> None:
        """Remove every ``.input_log`` directory the purge scope covers."""
        async with self._lock:
            await asyncio.to_thread(self._purge_scope_sync, scope)

    def _purge_scope_sync(self, scope: Any) -> None:
        if scope.kind == "space":
            dirs = list(self._root.glob("*/*/.input_log"))
        elif scope.kind == "app":
            dirs = list((self._root / str(scope.app_dir)).glob("*/.input_log"))
        else:
            dirs = [self._root / str(scope.app_dir) / str(scope.project_dir) / _LOG_DIR]
        for directory in dirs:
            try:
                shutil.rmtree(directory)
            except FileNotFoundError:
                # Already gone is success; any other removal failure must
                # escape so the purge runner retries the step.
                continue

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


def _encode(line: dict[str, Any]) -> str:
    return json.dumps(line, ensure_ascii=False, separators=(",", ":"))
