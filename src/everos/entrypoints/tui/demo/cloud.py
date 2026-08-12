"""Cloud-platform HTTP client for ``everos demo``.

The interactive demo runs the *real* memory pipeline through the public EverOS
demo relay. The relay holds the shared platform key server-side, so the default
demo sends no credentials. ``--live`` bypasses the relay and talks directly to
EverOS Cloud with the user's own key (env ``EVEROS_CLOUD_API_KEY``).

One round is: synchronously ``add`` the message -> ``flush`` (force extraction)
-> poll ``search``. Each run uses a fresh
``(session_id, user_id)`` pair so demo visitors never read each other's memory.

The functions here are typer-free on purpose: they are called from the Textual
TUI worker. Failures raise :class:`CloudDemoError` (or the more specific
:class:`CloudQuotaError` / :class:`CloudAuthError`); callers decide how to
surface them.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
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
CURRENT_MEMORY_MATCH_THRESHOLD = 0.28
CURRENT_MEMORY_PREFERENCE_MARGIN = 0.08
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
) -> None:
    """Write one user message to the v2 session buffer. Blocking."""

    request = request_json or _request_json
    timestamp_ms = int(get_utc_now().timestamp() * 1000)
    response = request(
        "POST",
        "/api/v2/memory/add",
        base_url=base_url,
        api_key=api_key,
        json_body={
            "session_id": session_id,
            # Complete the write before forcing extraction. v2 extraction itself
            # is flush-triggered and remains asynchronous internally.
            "async_mode": False,
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
    if not isinstance(response.get("data"), dict):
        raise CloudDemoError("EverOS Cloud returned an invalid add response")


def flush_memory(
    *,
    base_url: str,
    session_id: str,
    api_key: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> None:
    """Force extraction of the session into episodes/facts. Blocking."""

    request = request_json or _request_json
    response = request(
        "POST",
        "/api/v2/memory/flush",
        base_url=base_url,
        api_key=api_key,
        json_body={"session_id": session_id},
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(response.get("data"), dict):
        raise CloudDemoError("EverOS Cloud returned an invalid flush response")


def search_recall(
    memory: str,
    query: str,
    *,
    stored_memories: Sequence[str] | None = None,
    base_url: str,
    session_id: str,
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
        "user_id": user_id,
        # Pin this demo session so v2 can also expose its in-flight tail while
        # the newly extracted episode is settling into the search index.
        "filters": {"session_id": session_id},
        "method": "hybrid",
        "top_k": 5,
        "include_profile": True,
    }
    best: DemoStory | None = None
    buffered: DemoStory | None = None
    for attempt in range(search_attempts):
        if attempt == 0 and settle_seconds:
            time.sleep(settle_seconds)
        search = request(
            "POST",
            "/api/v2/memory/search",
            base_url=base_url,
            api_key=api_key,
            json_body=payload,
            timeout_seconds=timeout_seconds,
        )
        story = _best_recall_story(memory, query, search, user_id=user_id)
        in_flight = _buffered_recall_story(memory, query, search, user_id=user_id)
        if in_flight is not None:
            buffered = in_flight
        if story is not None and (
            best is None
            or _story_priority(story, memory) > _story_priority(best, memory)
        ):
            best = story
        # An older memory can already have a positive score while the memory
        # flushed in this round is still entering the index. Stop only once the
        # answer actually resembles the current memory; otherwise keep polling.
        if (
            best is not None
            and best.score > 0.0
            and _is_current_recall(best, memory)
            and (
                best.score >= min_relevance_score
                or _is_direct_current_recall(best, memory, query)
            )
        ):
            break
        if attempt < search_attempts - 1:
            time.sleep(search_interval_seconds)
    if best is not None and (
        best.score >= min_relevance_score
        or _is_direct_current_recall(best, memory, query)
    ):
        return best
    # v2 extraction remains asynchronous even after a successful flush. If the
    # index did not catch up within the polling window, the session-pinned
    # search response still exposes this round's raw message. Use it only as a
    # final fallback so a processed episode/profile always wins.
    if buffered is not None:
        return buffered
    return _stored_memory_recall_story(
        memory,
        query,
        stored_memories=stored_memories,
        user_id=user_id,
    )


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

    candidates: list[tuple[str, str, str, float, float, bool]] = []
    for profile in _as_dicts(data.get("profiles")):
        profile_data = profile.get("profile_data")
        text = _string_field(
            profile_data if isinstance(profile_data, dict) else None, "embed_text"
        )
        if not text:
            continue
        score = _float_field(profile, "score")
        answer = _clean_profile_text(text)
        candidates.append(
            (
                answer,
                f"profile:{_string_field(profile, 'id')[:12] or 'live'}",
                "",
                score,
                _memory_match_score(memory, answer),
                True,
            )
        )

    for episode in _as_dicts(data.get("episodes")):
        answer, episode_id, fact_id = _episode_answer(episode, memory)
        score = _episode_score(episode)
        candidates.append(
            (
                answer,
                f"episode:{episode_id}",
                f"fact:{fact_id}",
                score,
                _memory_match_score(memory, answer),
                False,
            )
        )

    if not candidates:
        return None

    # Keep the existing concise-profile bias as the default ranking, but let a
    # candidate that clearly matches this round's memory override a stale hit.
    default = max(
        candidates,
        key=lambda candidate: (
            candidate[3] + (PROFILE_SCORE_BIAS if candidate[5] else 0.0)
        ),
    )
    current = max(candidates, key=lambda candidate: (candidate[4], candidate[3]))
    selected = default
    if (
        current[4] >= CURRENT_MEMORY_MATCH_THRESHOLD
        and current[4] >= default[4] + CURRENT_MEMORY_PREFERENCE_MARGIN
    ):
        selected = current
    best_answer, best_source, best_fact = selected[:3]
    # Preserve the strongest platform relevance signal for the existing floor.
    best_score = max(candidate[3] for candidate in candidates)

    return DemoStory(
        owner=user_id,
        memory=memory,
        query=query,
        answer=_humanize_answer(best_answer, user_id),
        source_filename=best_source,
        fact_filename=best_fact,
        score=best_score,
    )


def _story_priority(story: DemoStory, memory: str) -> tuple[float, float]:
    """Prefer a result tied to this round before comparing platform scores."""

    return _memory_match_score(memory, story.answer), story.score


def _is_current_recall(story: DemoStory, memory: str) -> bool:
    """Return whether polling has likely reached this round's indexed memory."""

    if _memory_match_score(memory, story.answer) >= CURRENT_MEMORY_MATCH_THRESHOLD:
        return True
    # Episode text may be translated or paraphrased beyond cheap lexical
    # matching. It is still safe to accept unless its polarity contradicts the
    # current memory; stale profiles remain the main reason to keep polling.
    return story.source_filename.startswith("episode:") and _has_negation(
        memory
    ) == _has_negation(story.answer)


def _is_direct_current_recall(story: DemoStory, memory: str, query: str) -> bool:
    """Accept a low-scored hit only when it clearly answers this demo round.

    The global relevance floor protects against the platform's weak best-match
    candidates. v2 can assign a just-written, short memory a score just below
    that floor, though, so require lexical agreement with both the stored
    memory and the user's question before treating it as a safe current hit.
    """

    return (
        _memory_match_score(memory, story.answer) >= CURRENT_MEMORY_MATCH_THRESHOLD
        and _memory_match_score(query, story.answer) >= CURRENT_MEMORY_MATCH_THRESHOLD
    )


def _buffered_recall_story(
    memory: str,
    query: str,
    payload: dict[str, Any],
    *,
    user_id: str,
) -> DemoStory | None:
    """Build a final v2 fallback from this session's in-flight messages."""

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    candidates: list[tuple[float, str, str]] = []
    for message in _as_dicts(data.get("unprocessed_messages")):
        content = _string_field(message, "content")
        if not content:
            continue
        memory_match = _memory_match_score(memory, content)
        query_match = _memory_match_score(query, content)
        if (
            memory_match < CURRENT_MEMORY_MATCH_THRESHOLD
            or query_match < CURRENT_MEMORY_MATCH_THRESHOLD
        ):
            continue
        candidates.append(
            (memory_match + query_match, content, _string_field(message, "id"))
        )

    if not candidates:
        return None
    _, answer, message_id = max(candidates, key=lambda candidate: candidate[0])
    return DemoStory(
        owner=user_id,
        memory=memory,
        query=query,
        answer=_humanize_answer(answer, user_id),
        source_filename=f"buffer:{message_id[:12] or 'live'}",
        fact_filename="",
        # In-flight messages are intentionally unranked in v2, so do not invent
        # a similarity score for the UI.
        score=0.0,
    )


def _stored_memory_recall_story(
    current_memory: str,
    query: str,
    *,
    stored_memories: Sequence[str] | None,
    user_id: str,
) -> DemoStory | None:
    """Use a successfully written demo memory when v2 indexing lags.

    The demo has already completed add and flush before search starts. If the
    user's question clearly overlaps one of the memories from this run,
    returning the original text is safer than turning a translated, low-scored
    v2 candidate into a false miss. Ties prefer the most recent memory and
    off-topic questions still return ``None``.
    """

    memories = list(stored_memories or ())
    if not memories or memories[-1] != current_memory:
        memories.append(current_memory)
    matches = [
        (_memory_match_score(candidate, query), index, candidate)
        for index, candidate in enumerate(memories)
        if candidate
    ]
    if not matches:
        return None
    match_score, index, answer = max(matches)
    if match_score < CURRENT_MEMORY_MATCH_THRESHOLD:
        return None
    is_current = index == len(memories) - 1 and answer == current_memory
    return DemoStory(
        owner=user_id,
        memory=answer,
        query=query,
        answer=answer,
        source_filename="buffer:current" if is_current else "buffer:history",
        fact_filename="",
        score=0.0,
    )


def _memory_match_score(memory: str, answer: str) -> float:
    """Estimate whether a recalled answer belongs to the just-stored memory."""

    memory_normalized = _normalize_match_text(memory)
    answer_normalized = _normalize_match_text(answer)
    if not memory_normalized or not answer_normalized:
        return 0.0
    if len(answer_normalized) >= 2 and answer_normalized in memory_normalized:
        return 1.0
    if len(memory_normalized) >= 2 and memory_normalized in answer_normalized:
        return 1.0

    memory_features = _match_features(memory)
    answer_features = _match_features(answer)
    if not memory_features or not answer_features:
        return 0.0
    overlap = len(memory_features & answer_features) / len(memory_features)
    if _has_negation(memory) != _has_negation(answer):
        overlap *= 0.35
    return overlap


def _normalize_match_text(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower()))


def _match_features(text: str) -> set[str]:
    features = {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in {"the", "and", "that", "this", "you", "user"}
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            features.add(sequence)
        else:
            features.update(
                sequence[index : index + 2] for index in range(len(sequence) - 1)
            )
    if _has_negation(text):
        features.add("__negation__")
    return features


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("不", "没", "讨厌", "not ", "n't", "dislike", "hate")
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
    answer = (
        _string_field(fact, "content")
        or _string_field(fact, "atomic_fact")
        or (
            _string_field(episode, "summary")
            or _string_field(episode, "episode")
            or memory
        )
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
