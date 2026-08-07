"""Cloud-platform HTTP client for ``everos demo``.

The interactive demo runs the *real* memory pipeline through the public EverOS
demo relay. The relay holds the shared platform key server-side, so the default
demo sends no credentials. ``--live`` bypasses the relay and talks directly to
EverOS Cloud with the user's own key (env ``EVEROS_CLOUD_API_KEY``).

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

CLOUD_PLATFORM_API_BASE_URL = "https://api.evermind.ai"
CLOUD_API_BASE_URL = "https://everosdemo.com"
CLOUD_DEMO_SERVER_URL_ENV = "EVEROS_CLOUD_DEMO_URL"
CLOUD_DEMO_KEY_ENV = "EVEROS_CLOUD_DEMO_KEY"
CLOUD_USER_KEY_ENV = "EVEROS_CLOUD_API_KEY"
# The public demo authenticates at the relay. Never ship its platform key in the
# client. The environment override remains useful for testing a direct endpoint.
DEFAULT_CLOUD_DEMO_KEY = ""

TIMEOUT_SECONDS = 15.0
TASK_ATTEMPTS = 12
TASK_INTERVAL_SECONDS = 1.0
SEARCH_ATTEMPTS = 8
SEARCH_INTERVAL_SECONDS = 1.5
# How far ahead an episode must score to beat a concise profile answer. Profiles
# read as a direct one-liner; episodes are verbose summaries. A small bias keeps
# answers concise on ties and near-ties without hiding a clearly-better episode.
PROFILE_SCORE_BIAS = 0.08
# Relevance floor. The platform always returns its best candidate, even for an
# unrelated query (short texts get ~0.4 similarity to everything), so without a
# cutoff "am I a programmer?" would surface whatever single memory exists. Below
# this score we report an honest miss instead of an absurd answer. Tuned from
# observed scores: clearly-irrelevant queries top out ~0.48, real hits >= 0.50.
MIN_RELEVANCE_SCORE = 0.5
# The just-flushed memory needs a moment to land in the index. Searching
# immediately returns a stale ranking (older memories that are already indexed),
# which is why a "store X then recall X" round could come back with an unrelated
# earlier memory. Let indexing settle before the first search.
SEARCH_SETTLE_SECONDS = 2.0


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


def resolve_live_base_url(server_url: str) -> str:
    """Use the platform for ``--live`` unless the user supplied an override."""

    if server_url != LIVE_DEMO_SERVER_URL:
        return server_url
    return CLOUD_PLATFORM_API_BASE_URL


def resolve_demo_key() -> str:
    """Return an optional direct-test key; the public relay needs no client key."""

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
    for _ in range(attempts):
        # Wait before each poll: the async task takes ~1-2s to register, so
        # polling immediately would just log a benign 404 on the platform side.
        time.sleep(interval_seconds)
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
    settle_seconds: float = SEARCH_SETTLE_SECONDS,
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> DemoStory | None:
    """Search the query, polling while indexing catches up.

    Returns a :class:`DemoStory` (with the real recall score) on a hit, or
    ``None`` on a miss. A miss means either the platform returned nothing or the
    best candidate scored below ``min_relevance_score`` — an honest "no match"
    beats surfacing an unrelated memory for an off-topic question. Blocking.

    The just-flushed memory takes a moment to index, so we settle first and then
    keep the best-scored result across attempts rather than returning the first
    (possibly stale) hit — otherwise "store X, recall X" can return an unrelated
    older memory that was already indexed.

    We pool the response's *profiles* and *episodes*: profiles are concise,
    answer-shaped facts that score well on natural-language questions, while
    episodes are the raw recalled memories. The highest-scored candidate wins.
    """

    request = request_json or _request_json
    payload = {
        "query": query,
        "filters": {"user_id": user_id},
        "method": "hybrid",
        "top_k": 5,
    }
    best: DemoStory | None = None
    for attempt in range(search_attempts):
        if attempt == 0 and settle_seconds:
            time.sleep(settle_seconds)
        search = request(
            "POST",
            "/api/v1/memories/search",
            base_url=base_url,
            api_key=api_key,
            json_body=payload,
            timeout_seconds=timeout_seconds,
        )
        story = _best_recall_story(memory, query, search, user_id=user_id)
        if story is not None and (best is None or story.score > best.score):
            best = story
        # Any positive score means indexing has produced a ranked result; stop
        # polling. Whether it counts as a hit is decided by the relevance floor.
        if best is not None and best.score > 0.0:
            break
        if attempt < search_attempts - 1:
            time.sleep(search_interval_seconds)
    if best is not None and best.score >= min_relevance_score:
        return best
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


def _best_recall_story(
    memory: str,
    query: str,
    payload: dict[str, Any],
    *,
    user_id: str,
) -> DemoStory | None:
    """Pick the single highest-scored recall candidate from a search response.

    Pools *profiles* (concise answer-shaped facts) and *episodes* (raw recalled
    memories); the platform does not pre-sort them, so we score every candidate
    and keep the best. Returns ``None`` when the response carries no candidates.
    """

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    profile_score = -1.0
    profile_answer = ""
    profile_source = ""
    for profile in _as_dicts(data.get("profiles")):
        profile_data = profile.get("profile_data")
        text = _string_field(
            profile_data if isinstance(profile_data, dict) else None, "embed_text"
        )
        if not text:
            continue
        score = _float_field(profile, "score")
        if score > profile_score:
            profile_score = score
            profile_answer = _clean_profile_text(text)
            profile_source = f"profile:{_string_field(profile, 'id')[:12] or 'live'}"

    episode_score = -1.0
    episode_answer = ""
    episode_source = ""
    episode_fact = ""
    for episode in _as_dicts(data.get("episodes")):
        answer, episode_id, fact_id = _episode_answer(episode, memory)
        score = _episode_score(episode)
        if score > episode_score:
            episode_score = score
            episode_answer = answer
            episode_source = f"episode:{episode_id}"
            episode_fact = f"fact:{fact_id}"

    # Prefer the concise profile answer unless an episode clearly out-scores it:
    # profiles read as a direct one-line answer, episodes as a verbose summary.
    use_profile = profile_answer and profile_score + PROFILE_SCORE_BIAS >= episode_score
    if use_profile:
        best_answer, best_source, best_fact = profile_answer, profile_source, ""
    elif episode_answer:
        best_answer, best_source, best_fact = (
            episode_answer,
            episode_source,
            episode_fact,
        )
    else:
        return None
    # Report the strongest signal so the relevance floor never demotes a real hit
    # to a miss just because we displayed the (near-tied) concise profile answer.
    best_score = max(profile_score, episode_score, 0.0)

    return DemoStory(
        owner=user_id,
        memory=memory,
        query=query,
        answer=_humanize_answer(best_answer, user_id),
        source_filename=best_source,
        fact_filename=best_fact,
        score=best_score,
    )


def _as_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _episode_score(episode: dict[str, Any]) -> float:
    """Relevance score for ranking: episode score, else its top fact's score."""

    score = _float_field(episode, "score")
    if score:
        return score
    facts = episode.get("atomic_facts")
    first_fact = facts[0] if isinstance(facts, list) and facts else None
    return _float_field(first_fact if isinstance(first_fact, dict) else None, "score")


def _episode_answer(episode: dict[str, Any], memory: str) -> tuple[str, str, str]:
    """Return ``(answer, episode_id, fact_id)`` for an episode candidate.

    Cloud puts the recalled content in ``atomic_fact`` (concise) and falls back
    to the episode summary; ``memory`` is the last resort.
    """

    facts = episode.get("atomic_facts")
    first_fact = facts[0] if isinstance(facts, list) and facts else None
    fact = first_fact if isinstance(first_fact, dict) else None
    answer = _string_field(fact, "atomic_fact") or (
        _string_field(episode, "summary") or _string_field(episode, "episode") or memory
    )
    episode_id = _string_field(episode, "id") or "live"
    return answer, episode_id, _string_field(fact, "id") or "live"


def _clean_profile_text(text: str) -> str:
    """Tidy a profile ``embed_text`` for display.

    Profiles arrive as ``"<category>: <value>"``. The category is metadata that
    reads as noise next to the recalled value, so drop a short leading label
    (half- or full-width colon) and keep the value.
    """

    for separator in (": ", "\uff1a"):
        head, sep, tail = text.partition(separator)
        if sep and tail.strip() and len(head.split()) <= 3:
            return tail.strip()
    return text.strip()


def _humanize_answer(answer: str, user_id: str) -> str:
    """Strip the synthetic demo user_id out of platform-generated summaries.

    The platform phrases summaries like "everos_demo_ab12 said ...". The raw id
    is noise in a demo, so swap it for "you".
    """

    return answer.replace(user_id, "you")


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
