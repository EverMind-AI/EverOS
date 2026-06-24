"""Hosted-demo HTTP client for ``everos demo``.

The default demo runs the *real* memory pipeline against a hosted EverOS server
that holds the LLM + embedding keys server-side, so a user experiences genuine
extraction and recall without configuring any keys locally. The keys never reach
the client; this module only speaks the public memory HTTP API.

Each demo run uses a fresh ``(session_id, user_id)`` pair (see
:func:`new_demo_identity`) so concurrent visitors on the shared hosted server
never read each other's memories.

The functions here are typer-free on purpose: they are called from the Textual
TUI worker, which must not depend on the CLI presentation layer. Failures raise
:class:`CloudDemoError` (or :class:`CloudQuotaError` for an exhausted free
quota), and callers decide how to surface them.
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

LIVE_DEMO_SERVER_URL = "http://127.0.0.1:8000"
LIVE_DEMO_SESSION_ID = "everos-demo-live"
LIVE_DEMO_USER_ID = "everos_demo_user"
LIVE_DEMO_APP_ID = "default"
LIVE_DEMO_PROJECT_ID = "default"
LIVE_DEMO_TIMEOUT_SECONDS = 10.0
LIVE_DEMO_SEARCH_ATTEMPTS = 6
LIVE_DEMO_SEARCH_INTERVAL_SECONDS = 0.5

# Hosted demo endpoint. Overridable via the env var so the URL is not hard-wired
# into releases; the default is a placeholder until the server is deployed.
CLOUD_DEMO_SERVER_URL_ENV = "EVEROS_CLOUD_DEMO_URL"
DEFAULT_CLOUD_DEMO_SERVER_URL = "https://demo.everos.evermind.ai"


class CloudDemoError(Exception):
    """A hosted demo round could not be completed."""


class CloudQuotaError(CloudDemoError):
    """The hosted demo server hit its free per-visitor quota (HTTP 429)."""


def resolve_cloud_base_url(server_url: str) -> str:
    """Pick the cloud endpoint: explicit --server-url wins, then env, then default."""

    if server_url != LIVE_DEMO_SERVER_URL:
        return server_url
    return os.environ.get(CLOUD_DEMO_SERVER_URL_ENV, DEFAULT_CLOUD_DEMO_SERVER_URL)


def new_demo_identity() -> tuple[str, str]:
    """Generate a unique ``(session_id, user_id)`` pair for one demo run."""

    token = uuid.uuid4().hex[:12]
    return f"everos-demo-{token}", f"everos_demo_{token}"


def recall_round(
    memory: str,
    query: str,
    *,
    base_url: str,
    session_id: str,
    user_id: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    timeout_seconds: float = LIVE_DEMO_TIMEOUT_SECONDS,
    search_attempts: int = LIVE_DEMO_SEARCH_ATTEMPTS,
    search_interval_seconds: float = LIVE_DEMO_SEARCH_INTERVAL_SECONDS,
) -> DemoStory:
    """Run one ``add -> flush -> search`` round against a live EverOS server.

    Blocking (uses ``urllib`` + ``time.sleep`` while indexing catches up); call
    it from a worker thread, never directly on an event loop.
    """

    request = request_json or _request_json
    health = request(
        "GET",
        "/health",
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    if health.get("status") != "ok":
        raise CloudDemoError(f"EverOS server at {base_url} is not healthy")

    timestamp_ms = int(get_utc_now().timestamp() * 1000)
    request(
        "POST",
        "/api/v1/memory/add",
        base_url=base_url,
        json_body={
            "session_id": session_id,
            "app_id": LIVE_DEMO_APP_ID,
            "project_id": LIVE_DEMO_PROJECT_ID,
            "messages": [
                {
                    "sender_id": user_id,
                    "role": "user",
                    "timestamp": timestamp_ms,
                    "content": memory,
                }
            ],
        },
        timeout_seconds=timeout_seconds,
    )
    request(
        "POST",
        "/api/v1/memory/flush",
        base_url=base_url,
        json_body={
            "session_id": session_id,
            "app_id": LIVE_DEMO_APP_ID,
            "project_id": LIVE_DEMO_PROJECT_ID,
        },
        timeout_seconds=timeout_seconds,
    )

    search_payload = {
        "user_id": user_id,
        "app_id": LIVE_DEMO_APP_ID,
        "project_id": LIVE_DEMO_PROJECT_ID,
        "query": query,
        "top_k": 5,
    }
    for attempt in range(search_attempts):
        search = request(
            "POST",
            "/api/v1/memory/search",
            base_url=base_url,
            json_body=search_payload,
            timeout_seconds=timeout_seconds,
        )
        episode = _first_live_episode(search)
        if episode is not None:
            return _story_from_live_episode(memory, query, episode, user_id=user_id)
        if attempt < search_attempts - 1:
            time.sleep(search_interval_seconds)

    raise CloudDemoError(
        "EverOS accepted the memory, but search did not return it yet. "
        "Try again once indexing catches up."
    )


def _request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    json_body: dict[str, object] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise CloudQuotaError(base_url) from exc
        raise CloudDemoError(
            f"EverOS server at {base_url} returned HTTP {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:
        raise CloudDemoError(f"Could not reach EverOS server at {base_url}.") from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise CloudDemoError(f"EverOS server returned non-object JSON: {url}")
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
    fact_id = _string_field(first_fact, "id") if isinstance(first_fact, dict) else ""
    answer = (
        _string_field(first_fact, "content") if isinstance(first_fact, dict) else ""
    )
    if not answer:
        answer = (
            _string_field(episode, "summary")
            or _string_field(episode, "episode")
            or memory
        )
    episode_id = _string_field(episode, "id") or "live"
    return DemoStory(
        owner=user_id,
        memory=memory,
        query=query,
        answer=answer,
        source_filename=f"episode:{episode_id}",
        fact_filename=f"fact:{fact_id or 'live'}",
    )


def _string_field(payload: dict[str, Any] | None, key: str) -> str:
    if payload is None:
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""
