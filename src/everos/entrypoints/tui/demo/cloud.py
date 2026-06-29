"""Cloud-platform HTTP client for ``everos demo``.

The interactive demo runs the *real* memory pipeline against the EverOS Cloud
platform (``https://api.evermind.ai``). The platform holds all model keys and
manages storage, so a user experiences genuine extraction and recall by passing
a single platform API key — no server to deploy, no DNS, no model keys locally.

Auth is ``Authorization: Bearer <api_key>``. The default demo uses a restricted
demo key (env ``EVEROS_CLOUD_DEMO_KEY``); ``--live`` uses the user's own
platform key (env ``EVEROS_CLOUD_API_KEY``).

One round is: ``add`` (async, returns a task) -> wait for the task -> ``flush``
(force extraction) -> poll ``search``. Each run uses a fresh
``(session_id, user_id)`` pair so demo visitors never read each other's memory.

The functions here are typer-free on purpose: they are called from the Textual
TUI worker. Failures raise :class:`CloudDemoError` (or the more specific
:class:`CloudQuotaError` / :class:`CloudAuthError`); callers decide how to
surface them.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from everos.component.utils.datetime import get_utc_now
from everos.entrypoints.tui.demo.data import DemoStory

# Sentinel default for the --server-url option; a different value means the user
# explicitly pointed the demo somewhere else.
LIVE_DEMO_SERVER_URL = "http://127.0.0.1:8000"
LIVE_DEMO_SESSION_ID = "everos-demo-live"
LIVE_DEMO_USER_ID = "everos_demo_user"

CLOUD_API_BASE_URL = "https://api.evermind.ai"
CLOUD_DEMO_SERVER_URL_ENV = "EVEROS_CLOUD_DEMO_URL"
CLOUD_DEMO_KEY_ENV = "EVEROS_CLOUD_DEMO_KEY"
CLOUD_USER_KEY_ENV = "EVEROS_CLOUD_API_KEY"
# Restricted, shippable demo key goes here once the platform issues one. Empty
# means the demo reads the key from the env var instead (and otherwise reports
# "not configured").
DEFAULT_CLOUD_DEMO_KEY = ""

TIMEOUT_SECONDS = 15.0
TASK_ATTEMPTS = 12
TASK_INTERVAL_SECONDS = 1.0
SEARCH_ATTEMPTS = 8
SEARCH_INTERVAL_SECONDS = 1.5


class CloudDemoError(Exception):
    """A cloud demo round could not be completed."""


class CloudQuotaError(CloudDemoError):
    """The platform hit a rate/quota limit (HTTP 429)."""


class CloudAuthError(CloudDemoError):
    """The platform rejected the API key (HTTP 401/403)."""


def resolve_cloud_base_url(server_url: str) -> str:
    """Pick the API endpoint: explicit --server-url wins, then env, then default."""

    if server_url != LIVE_DEMO_SERVER_URL:
        return server_url
    return os.environ.get(CLOUD_DEMO_SERVER_URL_ENV, CLOUD_API_BASE_URL)


def resolve_demo_key() -> str:
    """The restricted demo key: env override, else the shipped default."""

    return os.environ.get(CLOUD_DEMO_KEY_ENV, DEFAULT_CLOUD_DEMO_KEY)


def resolve_user_key() -> str:
    """The user's own platform key for --live (env only)."""

    return os.environ.get(CLOUD_USER_KEY_ENV, "")


def new_demo_identity() -> tuple[str, str]:
    """Generate a unique ``(session_id, user_id)`` pair for one demo run."""

    token = uuid.uuid4().hex[:12]
    return f"everos-demo-{token}", f"everos_demo_{token}"


def add_memory(
    memory: str,
    *,
    base_url: str,
    session_id: str,
    user_id: str,
    api_key: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> str:
    """Queue one user memory; returns the async task id. Blocking."""

    request = request_json or _request_json
    timestamp_ms = int(get_utc_now().timestamp() * 1000)
    resp = request(
        "POST",
        "/api/v1/memories",
        base_url=base_url,
        api_key=api_key,
        json_body={
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {"role": "user", "timestamp": timestamp_ms, "content": memory}
            ],
        },
        timeout_seconds=timeout_seconds,
    )
    data = resp.get("data")
    return _string_field(data if isinstance(data, dict) else None, "task_id")


def wait_task(
    task_id: str,
    *,
    base_url: str,
    api_key: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    attempts: int = TASK_ATTEMPTS,
    interval_seconds: float = TASK_INTERVAL_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> None:
    """Best-effort poll of the async add task until it finishes. Blocking.

    Transient errors (e.g. a 404 before the task registers) are tolerated; if
    the task never reports success we return anyway and let flush/search drive
    eventual consistency. A reported failure raises.
    """

    if not task_id:
        return
    request = request_json or _request_json
    for attempt in range(attempts):
        status = ""
        try:
            resp = request(
                "GET",
                f"/api/v1/tasks/{task_id}",
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            data = resp.get("data") if isinstance(resp.get("data"), dict) else resp
            status = _string_field(data, "status")
        except CloudQuotaError:
            raise
        except CloudDemoError:
            status = ""  # task not registered yet / transient
        if status in {"success", "completed", "done"}:
            return
        if status in {"failed", "error"}:
            raise CloudDemoError("memory processing failed")
        if attempt < attempts - 1:
            time.sleep(interval_seconds)


def flush_memory(
    *,
    base_url: str,
    session_id: str,
    user_id: str,
    api_key: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> None:
    """Force extraction of the session into episodes/facts. Blocking."""

    request = request_json or _request_json
    request(
        "POST",
        "/api/v1/memories/flush",
        base_url=base_url,
        api_key=api_key,
        json_body={"user_id": user_id, "session_id": session_id, "force": True},
        timeout_seconds=timeout_seconds,
    )


def search_recall(
    memory: str,
    query: str,
    *,
    base_url: str,
    user_id: str,
    api_key: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    search_attempts: int = SEARCH_ATTEMPTS,
    search_interval_seconds: float = SEARCH_INTERVAL_SECONDS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> DemoStory | None:
    """Search the query, polling while indexing catches up.

    Returns a :class:`DemoStory` (with the real recall score) on a hit, or
    ``None`` on a miss (the platform answered but returned nothing). Blocking.
    """

    request = request_json or _request_json
    payload = {
        "query": query,
        "filters": {"user_id": user_id},
        "method": "hybrid",
        "top_k": 5,
    }
    for attempt in range(search_attempts):
        search = request(
            "POST",
            "/api/v1/memories/search",
            base_url=base_url,
            api_key=api_key,
            json_body=payload,
            timeout_seconds=timeout_seconds,
        )
        episode = _first_live_episode(search)
        if episode is not None:
            return _story_from_live_episode(memory, query, episode, user_id=user_id)
        if attempt < search_attempts - 1:
            time.sleep(search_interval_seconds)
    return None


def _request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    api_key: str | None = None,
    json_body: dict[str, object] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudAuthError(
                "EverOS Cloud rejected the API key (set EVEROS_CLOUD_DEMO_KEY)."
            ) from exc
        if exc.code == 429:
            raise CloudQuotaError(base_url) from exc
        raise CloudDemoError(
            f"EverOS Cloud at {base_url} returned HTTP {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:
        raise CloudDemoError(f"Could not reach EverOS Cloud at {base_url}.") from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise CloudDemoError(f"EverOS Cloud returned non-object JSON: {url}")
    return parsed


def _first_live_episode(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    first = episodes[0]
    return first if isinstance(first, dict) else None


def _story_from_live_episode(
    memory: str,
    query: str,
    episode: dict[str, Any],
    *,
    user_id: str,
) -> DemoStory:
    facts = episode.get("atomic_facts")
    first_fact = facts[0] if isinstance(facts, list) and facts else None
    fact = first_fact if isinstance(first_fact, dict) else None
    fact_id = _string_field(fact, "id")
    # Cloud puts the recalled content in ``atomic_fact`` and the score on the
    # fact (episode-level score is null).
    answer = _string_field(fact, "atomic_fact") or (
        _string_field(episode, "summary")
        or _string_field(episode, "episode")
        or memory
    )
    score = _float_field(fact, "score") or _float_field(episode, "score")
    episode_id = _string_field(episode, "id") or "live"
    return DemoStory(
        owner=user_id,
        memory=memory,
        query=query,
        answer=answer,
        source_filename=f"episode:{episode_id}",
        fact_filename=f"fact:{fact_id or 'live'}",
        score=score,
    )


def _string_field(payload: dict[str, Any] | None, key: str) -> str:
    if payload is None:
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _float_field(payload: dict[str, Any] | None, key: str) -> float:
    if payload is None:
        return 0.0
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else 0.0
