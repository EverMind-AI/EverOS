"""Synchronous HTTP client for the EverOS memory API.

Wraps :class:`httpx.AsyncClient` running on a single dedicated background
event-loop thread (daemon) so the public surface stays synchronous — the
Hermes memory-provider ABC is sync, and this avoids blocking a global loop or
spawning a loop per call. All network / server errors are normalised to
:class:`EverOSClientError` with a machine-readable ``code`` field so the
provider's circuit breaker can classify transient vs client failures.

This module is Hermes-agnostic: it depends only on the stdlib, ``httpx``, and
its sibling ``_types`` / ``_constants`` modules. It must not import anything
from ``everos.*`` (this code runs inside the Hermes process).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

import httpx

from ._constants import _ADD_BATCH_SIZE, _DEFAULT_APP_ID, _DEFAULT_PROJECT_ID
from ._types import (
    AddResponse,
    EverOSClientError,
    FlushResponse,
    GetData,
    MessageItem,
    SearchData,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ADD_PATH = "/api/v1/memory/add"
_FLUSH_PATH = "/api/v1/memory/flush"
_SEARCH_PATH = "/api/v1/memory/search"
_GET_PATH = "/api/v1/memory/get"

# Future-result backstop: httpx (timeout=self._timeout) fires first; the extra
# grace ensures the mapped httpx.TimeoutException wins over a bare
# concurrent.futures.TimeoutError (which would leave a zombie coroutine).
_FUTURE_GRACE_SECS = 5.0

# Optional query params forwarded from **kwargs (None => omitted from body).
_SEARCH_KWARGS = (
    "method",
    "top_k",
    "radius",
    "min_score",
    "include_profile",
    "enable_llm_rerank",
    "filters",
)
_GET_KWARGS = ("page", "page_size", "sort_by", "sort_order", "filters")


class EverOSApiClient:
    """Synchronous client for the EverOS ``/api/v1/memory/*`` endpoints.

    The event-loop thread and ``httpx.AsyncClient`` are created lazily on the
    first request. Call :meth:`close` to tear them down.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: httpx.AsyncClient | None = None
        self._lock = threading.Lock()
        self._closed = False

    # ── loop / transport plumbing ───────────────────────────────────────────

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return
        with self._lock:
            # Re-check under the lock: close() may have nulled _loop (and set
            # _closed) between the outer check and acquiring the lock. A worker
            # that passed the _closed gate in _run() before close() ran must
            # not resurrect a fresh loop on a closed client.
            if self._closed:
                raise EverOSClientError("client is closed", code="CLIENT_CLOSED")
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name="everos-api-loop",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._thread = thread
            # Build the AsyncClient inside the loop thread so its async
            # lifecycle is bound to the loop that will use it.
            init = asyncio.run_coroutine_threadsafe(self._make_client(), loop)
            init.result()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _make_client(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Post ``coro`` to the background loop and wait for its result."""
        if self._closed:
            raise EverOSClientError("client is closed", code="CLIENT_CLOSED")
        self._ensure_loop()
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=self._timeout + _FUTURE_GRACE_SECS)
        except TimeoutError as exc:
            # Future did not resolve in time (loop stalled or request hung).
            raise EverOSClientError(
                "request timed out waiting for event loop",
                code="EXTERNAL_SERVICE_UNAVAILABLE",
            ) from exc

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        try:
            resp = await self._client.post(path, json=body)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EverOSClientError(
                str(exc) or exc.__class__.__name__,
                code="EXTERNAL_SERVICE_UNAVAILABLE",
            ) from exc
        if resp.status_code >= 400:
            self._raise_for_status(resp)
        try:
            envelope = resp.json()
        except ValueError as exc:
            raise EverOSClientError(
                f"HTTP {resp.status_code}: non-JSON response",
                code="INTERNAL_ERROR",
            ) from exc
        data = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(data, dict):
            raise EverOSClientError(
                "response envelope missing data object",
                code="INTERNAL_ERROR",
            )
        return data

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        try:
            envelope = resp.json()
        except ValueError:
            raise EverOSClientError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                code="INTERNAL_ERROR",
            ) from None
        error = envelope.get("error") if isinstance(envelope, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "INTERNAL_ERROR")
            message = str(error.get("message") or f"HTTP {resp.status_code}")
            raise EverOSClientError(message, code=code)
        raise EverOSClientError(
            f"HTTP {resp.status_code}: {resp.text[:200]}",
            code="INTERNAL_ERROR",
        )

    # ── scope / owner helpers ───────────────────────────────────────────────

    @staticmethod
    def _apply_scope(body: dict[str, Any], app_id: str, project_id: str) -> None:
        """Omit app_id/project_id when they equal the server default."""
        if app_id != _DEFAULT_APP_ID:
            body["app_id"] = app_id
        if project_id != _DEFAULT_PROJECT_ID:
            body["project_id"] = project_id

    @staticmethod
    def _owner_key(user_id: str | None, agent_id: str | None) -> str:
        """Return the owner field name to populate, enforcing XOR."""
        if (user_id is None) == (agent_id is None):
            raise ValueError(
                "exactly one of user_id / agent_id must be provided",
            )
        return "user_id" if user_id is not None else "agent_id"

    # ── public API ──────────────────────────────────────────────────────────

    def add_messages(
        self,
        session_id: str,
        app_id: str,
        project_id: str,
        messages: Sequence[MessageItem],
    ) -> AddResponse:
        """POST /api/v1/memory/add, chunking into batches of _ADD_BATCH_SIZE.

        Sends sequential requests and merges results: ``message_count`` is the
        sum across batches, ``status`` is the last batch's status.
        """
        body_base: dict[str, Any] = {"session_id": session_id}
        self._apply_scope(body_base, app_id, project_id)
        total = len(messages)
        message_count = 0
        status: str | None = None
        # max(total, 1) ensures an empty input still hits the server (which
        # rejects it with 422) rather than silently no-op'ing.
        for start in range(0, max(total, 1), _ADD_BATCH_SIZE):
            batch = list(messages[start : start + _ADD_BATCH_SIZE])
            body = {**body_base, "messages": batch}
            resp = self._run(self._post(_ADD_PATH, body))
            message_count += int(resp.get("message_count", 0))
            status = resp.get("status")
        assert status is not None
        return {"message_count": message_count, "status": status}  # type: ignore[return-value]

    def flush_session(
        self,
        session_id: str,
        app_id: str,
        project_id: str,
    ) -> FlushResponse:
        """POST /api/v1/memory/flush."""
        body: dict[str, Any] = {"session_id": session_id}
        self._apply_scope(body, app_id, project_id)
        return self._run(self._post(_FLUSH_PATH, body))  # type: ignore[return-value]

    def search(
        self,
        user_id: str | None,
        agent_id: str | None,
        app_id: str,
        project_id: str,
        query: str,
        **kwargs: Any,
    ) -> SearchData:
        """POST /api/v1/memory/search. Returns the ``data`` object.

        Exactly one of ``user_id`` / ``agent_id`` must be set. Accepted kwargs:
        method, top_k, radius, min_score, include_profile, enable_llm_rerank,
        filters. ``None``-valued kwargs are omitted so server defaults apply.
        """
        owner = self._owner_key(user_id, agent_id)
        body: dict[str, Any] = {
            "query": query,
            owner: user_id if owner == "user_id" else agent_id,
        }
        self._apply_scope(body, app_id, project_id)
        for key in _SEARCH_KWARGS:
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return self._run(self._post(_SEARCH_PATH, body))  # type: ignore[return-value]

    def get(
        self,
        user_id: str | None,
        agent_id: str | None,
        app_id: str,
        project_id: str,
        memory_type: str,
        **kwargs: Any,
    ) -> GetData:
        """POST /api/v1/memory/get. Returns the ``data`` object.

        Exactly one of ``user_id`` / ``agent_id`` must be set. Accepted kwargs:
        page, page_size, sort_by, sort_order, filters.
        """
        owner = self._owner_key(user_id, agent_id)
        body: dict[str, Any] = {
            "memory_type": memory_type,
            owner: user_id if owner == "user_id" else agent_id,
        }
        self._apply_scope(body, app_id, project_id)
        for key in _GET_KWARGS:
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return self._run(self._post(_GET_PATH, body))  # type: ignore[return-value]

    def close(self) -> None:
        """Shut down the loop thread and close the httpx client."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        client = self._client
        if loop is not None and client is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
                fut.result(timeout=10.0)
            except Exception:
                logger.debug("error closing httpx client", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._client = None
        self._loop = None
        self._thread = None
