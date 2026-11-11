"""End-to-end tests for ``POST /api/v2/memory/search``.

Multi-round retrieval (``llm_multiround``) is the only search method: each
round embeds its sub-queries, fuses BM25 and vector recall per sub-query, and
asks a decider which candidates to keep. White-box e2e: the real application
and route over real LanceDB rows reloaded from the
``tests/fixtures/search_seed/`` slice (16 episodes / 2 user_profiles produced
by ``_dump_search_seed.py``, with real vectors and real BM25 tokens).

The contract tests run on a fake 1024-wide embedder and a scripted decider
that keeps the first ten candidates (:func:`tests.fakes.memorize_llm`), so
they need no credentials and pin what the route promises whatever the
ranking:

- the default method, and the decider actually running
- owner isolation
- top_k (specific value + -1 unlimited)
- include_profile (true / false)
- filter DSL: session_id eq / timestamp range / sender_id in /
  parent_id (memcell) / top-level OR / nested AND-OR, and its 422
- the envelope (``episodes`` / ``images`` / ``profiles`` /
  ``unprocessed_messages`` / ``scope``)
- unprocessed_messages from the boundary-detection buffer
- timezone discipline across a display-tz switch

Request validation (the removed methods and their options, agent scopes) is
pinned by the unit suite's ``test_search_route_validation.py``; an agent's
cases and skills are listed by ``/memory/get`` (``test_get_endpoint_e2e.py``).

``test_multiround_recall_with_real_providers`` drives the same route with the
real embedder and decider from ``.env``: it carries ``@pytest.mark.live_llm``
and ``@pytest.mark.slow``, which CI's default
``-m 'not slow and not live_llm'`` deselects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient, Response

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import Episode, UserProfile
from everos.service import Providers, build_application
from tests.fakes import FakeEmbeddingProvider, make_providers, memorize_llm
from tests.fakes.storage import StorageView

_SEARCH = "/api/v2/memory/search"

# ── Fixtures: the application over the test's memory root, no lifespan ──


@asynccontextmanager
async def _serve(
    storage_view: StorageView, providers: Providers | None
) -> AsyncIterator[AsyncClient]:
    """An HTTP client on an unstarted application (no cascade, no OME).

    ``providers=None`` builds the providers ``.env`` configures.
    """
    # The application is never started, so create the SQLite tables its
    # ``start`` would; the search reads the boundary buffer from them.
    await storage_view.create_schema()
    application = build_application(load_settings(), providers=providers)
    app = create_app(application=application)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
    finally:
        await application.aclose()


@pytest.fixture
def llm_calls() -> list[str]:
    """Reply models the scripted model was asked for, in call order."""
    return []


@pytest.fixture
async def client(
    storage_view: StorageView, llm_calls: list[str]
) -> AsyncIterator[AsyncClient]:
    """The app on a fake 1024-wide embedder and a keep-the-first-ten decider."""
    providers = make_providers(
        llm=memorize_llm(calls=llm_calls),
        embedding=FakeEmbeddingProvider(dim=1024),
    )
    async with _serve(storage_view, providers) as c:
        yield c


@pytest.fixture
async def live_client(storage_view: StorageView) -> AsyncIterator[AsyncClient]:
    """The app on the real embedder and decider ``.env`` configures."""
    async with _serve(storage_view, None) as c:
        yield c


# ── Seed loading helpers ──────────────────────────────────────────────


async def _seed_episodes(
    storage_view: StorageView, rows: list[dict[str, Any]]
) -> list[Episode]:
    """Validate + add episode rows + build the FTS index.

    Tests should pre-mutate per-row fields (e.g. ``session_id``) in the
    list passed in, rather than calling this multiple times. LanceDB's
    FTS index covers rows present at the time of index creation; calling
    :func:`ensure_business_indexes` here rebuilds the index over the
    current row set so the BM25 recall can see them.
    """
    seeded = [Episode.model_validate(r) for r in rows]
    await storage_view.repos.episode.add(seeded)
    await storage_view.index.ensure_business_indexes()
    return seeded


async def _seed_user_profiles(
    storage_view: StorageView, rows: list[dict[str, Any]]
) -> list[UserProfile]:
    profiles = [UserProfile.model_validate(r) for r in rows]
    await storage_view.repos.user_profile.add(profiles)
    # profile table has no FTS — no index rebuild needed.
    return profiles


def _eps_for_owner(seed: dict, owner: str) -> list[dict]:
    return [r for r in seed["episode"] if r["owner_id"] == owner]


def _profiles_for_owner(seed: dict, owner: str) -> list[dict]:
    return [r for r in seed["user_profile"] if r["owner_id"] == owner]


def _post(
    client: AsyncClient,
    *,
    user_id: str = "caroline",
    query: str = "support",
    method: str | None = None,
    top_k: int | None = None,
    include_profile: bool | None = None,
    filters: dict | None = None,
) -> Awaitable[Response]:
    """Build a search request body; ``method`` is left to its default."""
    body: dict[str, Any] = {"user_id": user_id, "query": query}
    if method is not None:
        body["method"] = method
    if top_k is not None:
        body["top_k"] = top_k
    if include_profile is not None:
        body["include_profile"] = include_profile
    if filters is not None:
        body["filters"] = filters
    return client.post(_SEARCH, json=body, timeout=60.0)


# ═══════════════════════════════════════════════════════════════════════
# 1. Multi-round recall
# ═══════════════════════════════════════════════════════════════════════


async def test_search_recalls_the_owners_episodes(
    client: AsyncClient,
    search_seed: dict,
    storage_view: StorageView,
    llm_calls: list[str],
) -> None:
    """The default method runs the multi-round loop over the owner's episodes."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    resp = await _post(client, query="LGBTQ support group")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["episodes"]
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"
        assert ep["type"] == "Conversation"
        assert isinstance(ep["score"], float)
    # One-to-one memory: no pictures were seeded, no profile was asked for,
    # and a user search carries no group scope.
    assert data["images"] == []
    assert data["profiles"] == []
    assert data["scope"] is None
    # The decider ran: this is the multi-round loop, not a fallback.
    assert "_DecisionReply" in llm_calls


async def test_naming_the_method_runs_the_same_search(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``method="llm_multiround"`` is accepted and changes nothing."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    default = await _post(client, query="counseling")
    named = await _post(client, query="counseling", method="llm_multiround")
    assert default.status_code == named.status_code == 200, named.text
    default_ids = [ep["id"] for ep in default.json()["data"]["episodes"]]
    assert default_ids
    assert [ep["id"] for ep in named.json()["data"]["episodes"]] == default_ids


# ═══════════════════════════════════════════════════════════════════════
# 2. Owner isolation
# ═══════════════════════════════════════════════════════════════════════


async def test_search_owner_isolation(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Same query, two owners — each only sees their own episodes."""
    await _seed_episodes(
        storage_view,
        _eps_for_owner(search_seed, "caroline")
        + _eps_for_owner(search_seed, "melanie"),
    )

    resp_c = await _post(client, user_id="caroline", query="caroline")
    resp_m = await _post(client, user_id="melanie", query="caroline")
    assert resp_c.status_code == 200
    assert resp_m.status_code == 200

    c_ids = {ep["id"] for ep in resp_c.json()["data"]["episodes"]}
    m_ids = {ep["id"] for ep in resp_m.json()["data"]["episodes"]}
    assert c_ids and m_ids
    assert c_ids.isdisjoint(m_ids)
    assert all(ep["user_id"] == "caroline" for ep in resp_c.json()["data"]["episodes"])
    assert all(ep["user_id"] == "melanie" for ep in resp_m.json()["data"]["episodes"])


# ═══════════════════════════════════════════════════════════════════════
# 3. top_k
# ═══════════════════════════════════════════════════════════════════════


async def test_search_top_k_caps_result_count(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``top_k=2`` returns at most 2 memories."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))
    resp = await _post(client, query="caroline", top_k=2)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["episodes"]
    assert len(data["episodes"]) + len(data["images"]) <= 2


async def test_search_top_k_unlimited_returns_all_matching(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``top_k=-1`` (default) means unlimited, capped by what is stored."""
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(storage_view, eps)
    unlimited = await _post(client, query="caroline", top_k=-1)
    capped = await _post(client, query="caroline", top_k=2)
    assert unlimited.status_code == capped.status_code == 200
    n_unlimited = len(unlimited.json()["data"]["episodes"])
    # Caroline has 8 seeded episodes; unlimited returns up to that.
    assert len(capped.json()["data"]["episodes"]) <= n_unlimited <= len(eps)


# ═══════════════════════════════════════════════════════════════════════
# 4. include_profile
# ═══════════════════════════════════════════════════════════════════════


async def test_search_include_profile_false_returns_no_profile(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Default ``include_profile=False`` leaves ``data.profiles`` empty."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))
    await _seed_user_profiles(
        storage_view, _profiles_for_owner(search_seed, "caroline")
    )

    resp = await _post(client, query="LGBTQ")
    assert resp.status_code == 200
    assert resp.json()["data"]["profiles"] == []


async def test_search_include_profile_true_returns_profile(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``include_profile=true`` (user owner) attaches the profile row."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))
    await _seed_user_profiles(
        storage_view, _profiles_for_owner(search_seed, "caroline")
    )

    resp = await _post(client, query="LGBTQ", include_profile=True)
    assert resp.status_code == 200
    profiles = resp.json()["data"]["profiles"]
    assert len(profiles) == 1
    p = profiles[0]
    assert p["user_id"] == "caroline"
    # Direct fetch — no ranking, score is null.
    assert p["score"] is None
    # Profile data is unpacked from the row's json buckets.
    pd = p["profile_data"]
    assert "summary" in pd
    assert isinstance(pd.get("explicit_info"), list)
    assert isinstance(pd.get("implicit_traits"), list)


# ═══════════════════════════════════════════════════════════════════════
# 5. filter DSL (parity with /get's compile_filters_for_get)
# ═══════════════════════════════════════════════════════════════════════


async def test_search_filter_by_session_id(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``filters: {session_id: ...}`` confines the search to one session."""
    base = _eps_for_owner(search_seed, "caroline")
    # Re-tag half to a different session so the filter has something to do.
    half = len(base) // 2
    await _seed_episodes(
        storage_view,
        [{**r, "session_id": "sess_target"} for r in base[:half]]
        + [{**r, "session_id": "sess_other"} for r in base[half:]],
    )

    resp = await _post(client, query="caroline", filters={"session_id": "sess_target"})
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["session_id"] == "sess_target" for ep in eps)


async def test_search_filter_by_sender_id_in(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``filters: {sender_id: {in: [...]}}`` → array_has OR ... ."""
    base = _eps_for_owner(search_seed, "caroline")
    half = len(base) // 2
    await _seed_episodes(
        storage_view,
        [{**r, "sender_ids": ["alice", "assistant"]} for r in base[:half]]
        + [{**r, "sender_ids": ["bob", "assistant"]} for r in base[half:]],
    )

    resp = await _post(
        client, query="caroline", filters={"sender_id": {"in": ["alice"]}}
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all("alice" in ep["sender_ids"] for ep in eps)


async def test_search_filter_by_parent_id_memcell(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``filters: {parent_id: ...}`` — verifies memcell-based slicing."""
    base = _eps_for_owner(search_seed, "caroline")
    target_mc = base[0]["parent_id"]
    target_eps = [r for r in base if r["parent_id"] == target_mc]
    other_eps = [r for r in base if r["parent_id"] != target_mc]
    await _seed_episodes(storage_view, target_eps + other_eps)

    resp = await _post(
        client,
        query="caroline",
        filters={"parent_id": target_mc, "parent_type": "memcell"},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["id"] in {r["id"] for r in target_eps} for ep in eps)


async def test_search_filter_top_level_or(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Top-level ``OR`` combinator parses + executes correctly."""
    base = _eps_for_owner(search_seed, "caroline")
    third = len(base) // 3
    await _seed_episodes(
        storage_view,
        [{**r, "session_id": "sess_a"} for r in base[:third]]
        + [{**r, "session_id": "sess_b"} for r in base[third : third * 2]]
        + [{**r, "session_id": "sess_c"} for r in base[third * 2 :]],
    )

    resp = await _post(
        client,
        query="caroline",
        filters={"OR": [{"session_id": "sess_a"}, {"session_id": "sess_b"}]},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["session_id"] in {"sess_a", "sess_b"} for ep in eps)


async def test_search_filter_nested_and_or(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Nested ``AND`` inside ``OR`` exercises the recursive compiler."""
    base = _eps_for_owner(search_seed, "caroline")
    quarter = len(base) // 4
    built = (
        [
            {**r, "session_id": "sess_a", "parent_id": "mc_target"}
            for r in base[:quarter]
        ]
        + [
            {**r, "session_id": "sess_a", "parent_id": "mc_other"}
            for r in base[quarter : quarter * 2]
        ]
        + [
            {**r, "session_id": "sess_b", "parent_id": "mc_target"}
            for r in base[quarter * 2 : quarter * 3]
        ]
        + [
            {**r, "session_id": "sess_c", "parent_id": "mc_other"}
            for r in base[quarter * 3 :]
        ]
    )
    by_id = {r["id"]: r for r in built}
    await _seed_episodes(storage_view, built)

    resp = await _post(
        client,
        query="caroline",
        filters={
            "OR": [
                {
                    "AND": [
                        {"session_id": "sess_a"},
                        {"parent_id": "mc_target"},
                    ]
                },
                {"session_id": "sess_c"},
            ]
        },
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    # ``parent_id`` isn't surfaced in SearchEpisodeItem — cross-reference
    # back to the seed dict by id to assert the predicate held.
    for ep in eps:
        orig = by_id[ep["id"]]
        assert (
            orig["session_id"] == "sess_a" and orig["parent_id"] == "mc_target"
        ) or orig["session_id"] == "sess_c"


async def test_search_filter_by_timestamp_range(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``timestamp`` filter with the ``gte`` operator narrows by time.

    Caroline's 8 seed episodes span 2023-05-08 → 2023-07-15; the
    ``2023-07-01`` cut splits them into a "before" and "after" group.
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(storage_view, eps)

    cutoff = "2023-07-01T00:00:00"
    expected_after = {r["id"] for r in eps if r["timestamp"] >= cutoff}
    assert expected_after, "seed should have at least one episode after the cutoff"

    resp = await _post(
        client,
        query="counseling",
        top_k=-1,
        filters={"timestamp": {"gte": cutoff}},
    )
    assert resp.status_code == 200
    returned = {ep["id"] for ep in resp.json()["data"]["episodes"]}
    # Every returned episode must be on/after the cutoff.
    assert returned
    assert returned <= expected_after


async def test_search_filter_no_match_returns_empty(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """A filter that matches no row → 200 + episodes=[], not 404 / 422."""
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    resp = await _post(
        client,
        query="caroline",
        filters={"session_id": "sess_that_does_not_exist"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["episodes"] == []
    rid = body["request_id"]
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)


async def test_search_filter_error_returns_422(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Filter DSL violations (unknown field) → 422 via the route's handler.

    ``FilterNode`` is permissive at Pydantic-validation time
    (``extra="allow"``) so unknown keys reach :func:`compile_filters`,
    which raises :class:`FilterError`. The route catches that and
    converts it to a clean 422 (separate path from Pydantic 422).
    """
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    resp = await _post(
        client, query="caroline", filters={"this_field_does_not_exist": "x"}
    )
    assert resp.status_code == 422, resp.text
    # Project's global exception handler shapes errors as
    # ``{request_id, error: {code, message, timestamp, path}}`` (not
    # FastAPI's default ``{"detail": ...}``). The FilterError text
    # lands in ``error.message``.
    body = resp.json()
    assert body["error"]["code"] == "INVALID_INPUT"
    assert "this_field_does_not_exist" in body["error"]["message"]


# ═══════════════════════════════════════════════════════════════════════
# 6. Envelope shape
# ═══════════════════════════════════════════════════════════════════════


async def test_search_returns_empty_envelope_for_unknown_owner(
    client: AsyncClient,
) -> None:
    """Owner with no seeded rows → 200 with every array empty.

    Pins the envelope-shape invariant: ``data.{episodes, images, profiles,
    unprocessed_messages}`` always exist (``scope`` is null on a user
    search); an empty result is a successful response, not 404.
    """
    resp = await _post(client, user_id="ghost", query="anything")
    assert resp.status_code == 200
    body = resp.json()
    rid = body["request_id"]
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)
    assert body["data"] == {
        "episodes": [],
        "images": [],
        "profiles": [],
        "unprocessed_messages": [],
        "scope": None,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. unprocessed_messages — in-flight boundary-detection buffer
# ═══════════════════════════════════════════════════════════════════════
#
# Buffer rows have no owner attribution (boundary detection runs before
# owner inference). The /search route surfaces them ONLY when ``filters``
# carries a top-level ``session_id`` eq scalar — the only meaningful
# query dimension on the unattributed buffer. Compound shapes (AND / OR,
# operator maps) do NOT trigger the lookup. A user search sees a session's
# buffer only when the user took part in it (a buffered message of theirs, or
# memory of theirs from that session), so every buffer below holds one of
# caroline's messages.
#
# White-box: the helpers below write directly into the
# ``unprocessed_buffer`` SQLite table via the public repo (no /add
# round-trip, so the boundary detector never runs and the rows stay
# unprocessed). The track string is boundary detection's ``memorize`` track
# (the shared single-pass detector track).


async def _seed_unprocessed_buffer_rows(
    storage_view: StorageView,
    session_id: str,
    rows_spec: list[dict[str, Any]],
    *,
    app_id: str = "default",
    project_id: str = "default",
    track: str = "memorize",
) -> None:
    """Drop a list of rows into the unprocessed_buffer table via the public repo.

    Mirrors the shape the memorize pipeline writes: ``content_items_json``
    holds the single-text ContentItem; ``text`` holds the derived plain
    string. Each ``rows_spec`` entry is ``{"message_id", "sender_id",
    "text", "role"?, "timestamp_offset_seconds"?}``. Timestamps default
    to sequential offsets from ``now`` so the repo's ts-asc ordering is
    deterministic.

    Always writes the full (session, track) slice in a single
    :meth:`storage_view.sqlite.unprocessed_buffer.replace` call — re-reading detached
    SQLModel instances and rewriting them in a later session triggers
    silent insert dedup on the PK, so callers must batch their rows.
    """
    import datetime as _dt2
    import json

    from everos.core.utils.datetime import get_now_with_timezone
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
    )

    base_ts = get_now_with_timezone()
    rows: list[UnprocessedBuffer] = []
    for idx, spec in enumerate(rows_spec):
        ts = base_ts + _dt2.timedelta(seconds=spec.get("timestamp_offset_seconds", idx))
        rows.append(
            UnprocessedBuffer(
                message_id=spec["message_id"],
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
                track=track,
                sender_id=spec["sender_id"],
                sender_name=None,
                role=spec.get("role", "user"),
                timestamp=ts,
                content_items_json=json.dumps([{"type": "text", "text": spec["text"]}]),
                text=spec["text"],
                tool_calls_json=None,
                tool_call_id=None,
            )
        )
    await storage_view.sqlite.unprocessed_buffer.replace(
        session_id,
        track,
        rows,
        app_id=app_id,
        project_id=project_id,
    )


async def test_search_unprocessed_messages_returned_for_top_level_session_filter(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """``filters={"session_id": "<sid>"}`` surfaces the buffered messages.

    Writes two rows into the unprocessed_buffer (same session) and one
    into a different session, then queries with the target session as a
    top-level eq scalar. Only the two matching rows should come back —
    in timestamp order, with the single-text shorthand collapsed to a
    plain ``str``.

    White-box surface: ``storage_view.sqlite.unprocessed_buffer`` (read & write via the
    same public repo the pipeline uses).
    """
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    # Two rows in the target session, one in a noise session.
    await _seed_unprocessed_buffer_rows(
        storage_view,
        "sess_inflight",
        [
            {
                "message_id": "msg_buf_001",
                "sender_id": "caroline",
                "text": "hello from buffer 1",
            },
            {
                "message_id": "msg_buf_002",
                "sender_id": "caroline",
                "text": "hello from buffer 2",
            },
        ],
    )
    await _seed_unprocessed_buffer_rows(
        storage_view,
        "sess_other",
        [
            {
                "message_id": "msg_buf_other",
                "sender_id": "caroline",
                "text": "should not appear",
            },
        ],
    )

    resp = await _post(
        client, query="anything", filters={"session_id": "sess_inflight"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    unprocessed = data["unprocessed_messages"]
    ids = [m["id"] for m in unprocessed]
    assert ids == ["msg_buf_001", "msg_buf_002"], (
        f"only target-session rows in ts order; got {ids}"
    )

    first = unprocessed[0]
    assert first["session_id"] == "sess_inflight"
    assert first["sender_id"] == "caroline"
    assert first["role"] == "user"
    # Single-text item collapses to the shorthand str.
    assert first["content"] == "hello from buffer 1"
    assert first["tool_calls"] is None
    assert first["tool_call_id"] is None


async def test_search_unprocessed_messages_empty_without_top_level_session_filter(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """No ``filters.session_id`` (or compound shape) → ``unprocessed_messages=[]``.

    Verifies the trigger semantics: the same buffer rows that the sibling
    test surfaces stay invisible to:

    1. a request without any ``filters``,
    2. a request whose ``session_id`` sits inside ``AND`` (not top-level),
    3. a request whose ``session_id`` uses the ``{"eq": ...}`` operator map.

    All three return ``[]`` even though matching buffer rows exist.

    White-box surface: ``storage_view.sqlite.unprocessed_buffer``.
    """
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))
    await _seed_unprocessed_buffer_rows(
        storage_view,
        "sess_inflight",
        [
            {
                "message_id": "msg_buf_xyz",
                "sender_id": "caroline",
                "text": "should never leak",
            },
        ],
    )

    # 1. No filters at all.
    resp_no_filter = await _post(client, query="anything")
    assert resp_no_filter.status_code == 200, resp_no_filter.text
    assert resp_no_filter.json()["data"]["unprocessed_messages"] == []

    # 2. session_id inside AND — compound, not top-level eq scalar.
    resp_and = await _post(
        client,
        query="anything",
        filters={"AND": [{"session_id": "sess_inflight"}]},
    )
    assert resp_and.status_code == 200, resp_and.text
    assert resp_and.json()["data"]["unprocessed_messages"] == []

    # 3. session_id via operator map — also not a top-level eq scalar.
    resp_op = await _post(
        client,
        query="anything",
        filters={"session_id": {"eq": "sess_inflight"}},
    )
    assert resp_op.status_code == 200, resp_op.text
    assert resp_op.json()["data"]["unprocessed_messages"] == []


async def test_search_unprocessed_messages_assistant_role_round_trips(
    client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """Buffer surfacing carries the original ``role`` through the wire.

    Pins that role / content shape survive the round trip — the buffered
    reply was inserted with ``role="assistant"`` and a multi-content payload
    would degrade to the opaque ``list[dict]`` shorthand, while a single
    text item collapses to ``str``.
    """
    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))
    await _seed_unprocessed_buffer_rows(
        storage_view,
        "sess_assistant",
        [
            {
                "message_id": "msg_buf_user",
                "sender_id": "caroline",
                "text": "what was the name of that museum?",
            },
            {
                "message_id": "msg_buf_asst",
                "sender_id": "bot",
                "text": "assistant-side buffered reply",
                "role": "assistant",
            },
        ],
    )

    resp = await _post(
        client, query="anything", filters={"session_id": "sess_assistant"}
    )
    assert resp.status_code == 200, resp.text
    msgs = resp.json()["data"]["unprocessed_messages"]
    assert [m["id"] for m in msgs] == ["msg_buf_user", "msg_buf_asst"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["sender_id"] == "bot"
    assert msgs[1]["content"] == "assistant-side buffered reply"


# ═══════════════════════════════════════════════════════════════════════
# 8. Timezone discipline — storage UTC vs display tz, no drift on switch
# ═══════════════════════════════════════════════════════════════════════
#
# Q2's core promise: a row written under ``EVEROS_MEMORY__TIMEZONE=X``
# read back under ``EVEROS_MEMORY__TIMEZONE=Y`` represents the **same
# UTC instant** — only the rendered offset changes. Without the
# storage-UTC discipline (Q2), a tz switch would silently misalign
# every stored row.
#
# Tests below exercise the full HTTP read path under two display zones
# back-to-back against the same on-disk row, plus a white-box probe
# into the SQLite ``unprocessed_buffer`` row to confirm the stored
# datetime is UTC-aware regardless of which zone the writer ran under.


async def _switch_display_tz(monkeypatch: pytest.MonkeyPatch, tz: str) -> None:
    """Switch ``EVEROS_MEMORY__TIMEZONE`` mid-test + drop both caches.

    Both ``load_settings`` and ``_display_tz`` are functools-cached;
    failing to clear them means the env-var change has no effect — see
    [docs/datetime.md](../../docs/datetime.md) §testing.
    """
    from everos.config import load_settings as _ls
    from everos.core.utils import datetime as dt_module

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", tz)
    _ls.cache_clear()
    dt_module._display_tz.cache_clear()


async def test_timezone_switch_preserves_utc_instant_across_responses(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
    storage_view: StorageView,
) -> None:
    """Q2 anti-drift contract: same on-disk row, two display zones, one instant.

    Real data path:
      1. Configure ``EVEROS_MEMORY__TIMEZONE=Asia/Shanghai`` and write
         an unprocessed-buffer row at a fixed instant.
      2. HTTP ``/search`` it: expect ``+08:00`` offset in the response.
      3. White-box probe SQLite directly: stored datetime is UTC-aware,
         not naive Shanghai bytes.
      4. Switch to ``EVEROS_MEMORY__TIMEZONE=UTC``, drop caches.
      5. HTTP ``/search`` the *same* row: expect ``Z`` / ``+00:00`` offset.
      6. Parse both response timestamps back to UTC instants — they
         must be equal **and** equal the original input instant.

    A pre-Q2 implementation would fail at step 5: the SQLite row stored
    naive Shanghai bytes, reinterpreted as UTC, would jump 8h forward.
    """
    import datetime as _real_dt
    import json as _json

    from everos.core.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
    )

    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    # Step 1: switch to Shanghai, write a row at a deterministic instant.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")

    # 06:00 UTC ↔ 14:00 Shanghai — pinned so the cross-zone assertion has
    # a clean number to read.
    target_instant_utc = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    row = UnprocessedBuffer(
        message_id="msg_tz_001",
        app_id="default",
        project_id="default",
        session_id="sess_tz_drift",
        track="memorize",
        sender_id="caroline",
        sender_name=None,
        role="user",
        timestamp=target_instant_utc,
        content_items_json=_json.dumps(
            [{"type": "text", "text": "deterministic-instant row"}]
        ),
        text="deterministic-instant row",
        tool_calls_json=None,
        tool_call_id=None,
    )
    await storage_view.sqlite.unprocessed_buffer.replace(
        "sess_tz_drift",
        "memorize",
        [row],
        app_id="default",
        project_id="default",
    )

    # Step 2: /search under Shanghai display tz → expect +08:00 offset.
    resp_sh = await _post(
        client, query="anything", filters={"session_id": "sess_tz_drift"}
    )
    assert resp_sh.status_code == 200, resp_sh.text
    msgs_sh = resp_sh.json()["data"]["unprocessed_messages"]
    assert len(msgs_sh) == 1
    ts_sh = msgs_sh[0]["timestamp"]
    assert ts_sh.endswith("+08:00"), f"expected Shanghai offset, got {ts_sh!r}"
    # 06:00 UTC rendered in +08:00 should read 14:00 local.
    assert "T14:00:00" in ts_sh, ts_sh

    # Step 3: white-box probe — SQLite row comes back UTC-aware.
    #
    # The SQLAlchemy ``load`` event listener registered on ``BaseTable``
    # (see ``infra/persistence/sqlite/_base/base.py``) re-attaches ``tzinfo=UTC``
    # to every ``UtcDatetime`` column after ORM hydrate, so callers never
    # observe a naive datetime — even though SQLite physically strips tz
    # at write time and SQLAlchemy ORM bypasses Pydantic on read.
    rows_back = await storage_view.sqlite.unprocessed_buffer.list_for_track(
        "sess_tz_drift", "memorize", app_id="default", project_id="default"
    )
    stored = rows_back[0].timestamp
    assert stored.tzinfo is _real_dt.UTC, (
        "SQLAlchemy load event hook should attach UTC on hydrate; "
        f"got tzinfo={stored.tzinfo!r}"
    )
    assert stored == target_instant_utc

    # Step 4: switch display tz to UTC, drop caches.
    await _switch_display_tz(monkeypatch, "UTC")

    # Step 5: /search the same row under UTC display tz.
    resp_utc = await _post(
        client, query="anything", filters={"session_id": "sess_tz_drift"}
    )
    assert resp_utc.status_code == 200, resp_utc.text
    msgs_utc = resp_utc.json()["data"]["unprocessed_messages"]
    assert len(msgs_utc) == 1
    ts_utc = msgs_utc[0]["timestamp"]
    # Pydantic canonicalises ``timezone.utc`` to the ``Z`` suffix.
    assert ts_utc.endswith("Z") or ts_utc.endswith("+00:00"), ts_utc
    # 06:00 UTC rendered with UTC display = 06:00 wall-clock.
    assert "T06:00:00" in ts_utc, ts_utc

    # Step 6: the anti-drift assertion — same UTC instant across both renders.
    instant_via_sh = from_iso_format(ts_sh).astimezone(_real_dt.UTC)
    instant_via_utc = from_iso_format(ts_utc).astimezone(_real_dt.UTC)
    assert instant_via_sh == instant_via_utc == target_instant_utc, (
        f"display-tz switch must not drift the UTC instant; "
        f"got {instant_via_sh=} {instant_via_utc=} {target_instant_utc=}"
    )


async def test_timezone_switch_preserves_utc_instant_for_episode(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
    storage_view: StorageView,
) -> None:
    """Same anti-drift contract, but for the LanceDB-backed episode path.

    LanceDB's Arrow schema declares timestamp columns with ``tz=UTC``
    (see ``lancedb.base.arrow_schema_for``) so PyArrow returns
    aware UTC datetimes directly — no caller-side ``ensure_utc`` needed.
    The shaper only runs ``to_display_tz`` to convert UTC to the
    configured display zone. This test pins that path against a tz switch.
    """
    import datetime as _real_dt

    from everos.core.utils.datetime import from_iso_format

    # Seed one episode at a fixed instant. The seed's ``timestamp`` is
    # passed straight into ``Episode.model_validate`` — LanceDB will
    # strip the tz at write time and store UTC bytes.
    pinned_utc = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    base = _eps_for_owner(search_seed, "caroline")[:1]
    base[0] = {**base[0], "timestamp": pinned_utc, "session_id": "sess_ep_tz"}
    await _seed_episodes(storage_view, base)

    # Step 1: Shanghai display tz.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(
        client, query="caroline", filters={"session_id": "sess_ep_tz"}
    )
    assert resp_sh.status_code == 200, resp_sh.text
    eps_sh = resp_sh.json()["data"]["episodes"]
    assert eps_sh, "the seeded episode should be recalled"
    ts_sh = eps_sh[0]["timestamp"]
    assert ts_sh.endswith("+08:00"), ts_sh
    assert "T14:00:00" in ts_sh, ts_sh

    # Step 2: switch to UTC.
    await _switch_display_tz(monkeypatch, "UTC")
    resp_utc = await _post(
        client, query="caroline", filters={"session_id": "sess_ep_tz"}
    )
    assert resp_utc.status_code == 200, resp_utc.text
    eps_utc = resp_utc.json()["data"]["episodes"]
    assert eps_utc
    ts_utc = eps_utc[0]["timestamp"]
    assert ts_utc.endswith("Z") or ts_utc.endswith("+00:00"), ts_utc
    assert "T06:00:00" in ts_utc, ts_utc

    # Step 3: anti-drift — same UTC instant before and after the switch.
    instant_via_sh = from_iso_format(ts_sh).astimezone(_real_dt.UTC)
    instant_via_utc = from_iso_format(ts_utc).astimezone(_real_dt.UTC)
    assert instant_via_sh == instant_via_utc == pinned_utc, (
        instant_via_sh,
        instant_via_utc,
        pinned_utc,
    )


async def test_timezone_reverse_switch_utc_to_shanghai(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
    storage_view: StorageView,
) -> None:
    """Reverse direction: write under UTC, read under Shanghai → no drift.

    Symmetric counterpart of
    :func:`test_timezone_switch_preserves_utc_instant_across_responses`.
    Covers the migration scenario where the default UTC deployment
    later turns on a local display tz.
    """
    import datetime as _real_dt
    import json as _json

    from everos.core.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
    )

    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    # Step 1: UTC display tz first, write at 06:00 UTC = 14:00 Shanghai.
    await _switch_display_tz(monkeypatch, "UTC")

    target_instant_utc = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    row = UnprocessedBuffer(
        message_id="msg_rev_001",
        app_id="default",
        project_id="default",
        session_id="sess_rev",
        track="memorize",
        sender_id="caroline",
        sender_name=None,
        role="user",
        timestamp=target_instant_utc,
        content_items_json=_json.dumps([{"type": "text", "text": "reverse-tz row"}]),
        text="reverse-tz row",
        tool_calls_json=None,
        tool_call_id=None,
    )
    await storage_view.sqlite.unprocessed_buffer.replace(
        "sess_rev",
        "memorize",
        [row],
        app_id="default",
        project_id="default",
    )

    # Step 2: read under UTC → expect Z / +00:00 + 06:00 wall clock.
    resp_utc = await _post(client, query="anything", filters={"session_id": "sess_rev"})
    assert resp_utc.status_code == 200, resp_utc.text
    msgs_utc = resp_utc.json()["data"]["unprocessed_messages"]
    ts_utc = msgs_utc[0]["timestamp"]
    assert ts_utc.endswith("Z") or ts_utc.endswith("+00:00"), ts_utc
    assert "T06:00:00" in ts_utc, ts_utc

    # Step 3: switch to Shanghai, read again → expect +08:00 + 14:00 wall clock.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(client, query="anything", filters={"session_id": "sess_rev"})
    assert resp_sh.status_code == 200, resp_sh.text
    msgs_sh = resp_sh.json()["data"]["unprocessed_messages"]
    ts_sh = msgs_sh[0]["timestamp"]
    assert ts_sh.endswith("+08:00"), ts_sh
    assert "T14:00:00" in ts_sh, ts_sh

    # Step 4: anti-drift — same UTC instant under both renders.
    instant_via_utc = from_iso_format(ts_utc).astimezone(_real_dt.UTC)
    instant_via_sh = from_iso_format(ts_sh).astimezone(_real_dt.UTC)
    assert instant_via_utc == instant_via_sh == target_instant_utc


async def test_timezone_mixed_timeline_two_writes_across_switch(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
    storage_view: StorageView,
) -> None:
    """Mixed timeline: write A under Shanghai, switch to UTC, write B, query both.

    Pins that storage UTC normalisation is **per-write** — switching
    display tz between writes does not corrupt earlier rows nor leak
    display-tz bytes into new rows. Both rows must come back as their
    original UTC instants regardless of which display tz the reader
    happens to be on.
    """
    import datetime as _real_dt
    import json as _json

    from everos.core.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
    )

    await _seed_episodes(storage_view, _eps_for_owner(search_seed, "caroline"))

    # Step 1: write row A under Shanghai.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    instant_a = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    row_a = UnprocessedBuffer(
        message_id="msg_mixed_a",
        app_id="default",
        project_id="default",
        session_id="sess_mixed",
        track="memorize",
        sender_id="caroline",
        sender_name=None,
        role="user",
        timestamp=instant_a,
        content_items_json=_json.dumps([{"type": "text", "text": "row A"}]),
        text="row A",
        tool_calls_json=None,
        tool_call_id=None,
    )

    # Step 2: switch to UTC, build row B.
    await _switch_display_tz(monkeypatch, "UTC")
    instant_b = _real_dt.datetime(2026, 5, 30, 18, 0, 0, tzinfo=_real_dt.UTC)
    row_b = UnprocessedBuffer(
        message_id="msg_mixed_b",
        app_id="default",
        project_id="default",
        session_id="sess_mixed",
        track="memorize",
        sender_id="caroline",
        sender_name=None,
        role="user",
        timestamp=instant_b,
        content_items_json=_json.dumps([{"type": "text", "text": "row B"}]),
        text="row B",
        tool_calls_json=None,
        tool_call_id=None,
    )

    # Persist both in one replace so the (session, track) slice owns both rows.
    # ``replace`` deletes the slice then inserts; a write-A/switch/write-B
    # interleaving on the wire would have the same storage effect.
    await storage_view.sqlite.unprocessed_buffer.replace(
        "sess_mixed",
        "memorize",
        [row_a, row_b],
        app_id="default",
        project_id="default",
    )

    # Step 3: query under Shanghai → both rows render +08:00, instants preserved.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(
        client, query="anything", filters={"session_id": "sess_mixed"}
    )
    assert resp_sh.status_code == 200, resp_sh.text
    msgs_sh = resp_sh.json()["data"]["unprocessed_messages"]
    by_id_sh = {m["id"]: m["timestamp"] for m in msgs_sh}
    assert by_id_sh.keys() == {"msg_mixed_a", "msg_mixed_b"}

    for ts in by_id_sh.values():
        assert ts.endswith("+08:00"), ts

    instant_a_sh = from_iso_format(by_id_sh["msg_mixed_a"]).astimezone(_real_dt.UTC)
    instant_b_sh = from_iso_format(by_id_sh["msg_mixed_b"]).astimezone(_real_dt.UTC)
    assert instant_a_sh == instant_a
    assert instant_b_sh == instant_b

    # Step 4: query under UTC → both render Z/+00:00, instants preserved.
    await _switch_display_tz(monkeypatch, "UTC")
    resp_utc = await _post(
        client, query="anything", filters={"session_id": "sess_mixed"}
    )
    assert resp_utc.status_code == 200, resp_utc.text
    msgs_utc = resp_utc.json()["data"]["unprocessed_messages"]
    by_id_utc = {m["id"]: m["timestamp"] for m in msgs_utc}

    for ts in by_id_utc.values():
        assert ts.endswith("Z") or ts.endswith("+00:00"), ts

    instant_a_utc = from_iso_format(by_id_utc["msg_mixed_a"]).astimezone(_real_dt.UTC)
    instant_b_utc = from_iso_format(by_id_utc["msg_mixed_b"]).astimezone(_real_dt.UTC)
    assert instant_a_utc == instant_a
    assert instant_b_utc == instant_b


# ═══════════════════════════════════════════════════════════════════════
# 9. Real providers — dense recall on the seed's real vectors
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
@pytest.mark.live_llm
async def test_multiround_recall_with_real_providers(
    live_client: AsyncClient, search_seed: dict, storage_view: StorageView
) -> None:
    """The real embedder and decider recall the owner's episodes, filtered.

    The seed's vectors come from the real embedder, so here the query's
    vector is comparable to them and the decider reads real candidates.
    Caroline's July 6 episode is about counseling and mental health; the
    ``2023-07-01`` cut keeps it and drops her earlier episodes.
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(storage_view, eps + _eps_for_owner(search_seed, "melanie"))

    cutoff = "2023-07-01T00:00:00"
    expected_after = {r["id"] for r in eps if r["timestamp"] >= cutoff}

    resp = await live_client.post(
        _SEARCH,
        json={
            "user_id": "caroline",
            "query": "How is Caroline looking after her mental health?",
            "method": "llm_multiround",
            "top_k": 5,
            "filters": {"timestamp": {"gte": cutoff}},
        },
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    returned = resp.json()["data"]["episodes"]
    assert returned
    assert all(ep["user_id"] == "caroline" for ep in returned)
    assert {ep["id"] for ep in returned} <= expected_after
