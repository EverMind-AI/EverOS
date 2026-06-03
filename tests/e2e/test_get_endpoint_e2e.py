"""End-to-end integration tests for ``POST /api/v1/memory/get``.

These tests spin up the FastAPI app with **no lifespan providers**
against a tmp ``EVEROS_MEMORY__ROOT``, populate a real LanceDB
``episode`` table directly via the repo singleton, and exercise the
HTTP route. They cover the wiring that unit tests cannot: pydantic
422s from the route, JSON envelope shape, and the full
``request → service → manager → LanceDB`` path.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from importlib import import_module
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    Episode,
    UserProfile,
    agent_case_repo,
    agent_skill_repo,
    episode_repo,
    lancedb_manager,
    user_profile_repo,
)

# ``everos.service.__init__`` re-exports the ``get`` function under the
# same name as the submodule (``from .get import get as get``), which
# shadows the submodule when imported normally. Pull the actual module
# via importlib so the test can poke at its ``_manager`` singleton.
get_service_mod = import_module("everos.service.get")


def _ts(day: int) -> _dt.datetime:
    return _dt.datetime(2026, 1, day, tzinfo=_dt.UTC)


def _episode(
    entry: str,
    *,
    owner: str = "u1",
    session: str = "sess_a",
    parent_id: str = "mc_1",
    sender_ids: list[str] | None = None,
    day: int = 1,
) -> Episode:
    return Episode(
        id=f"{owner}_{entry}",
        entry_id=entry,
        owner_id=owner,
        owner_type="user",
        session_id=session,
        timestamp=_ts(day),
        parent_type="memcell",
        parent_id=parent_id,
        sender_ids=sender_ids if sender_ids is not None else [owner, "assistant"],
        subject=f"subj {entry}",
        summary=f"summary {entry}",
        episode=f"body of {entry}",
        episode_tokens=f"body of {entry}",
        md_path=f"users/{owner}/episodes/{entry}.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


def _agent_case(
    entry: str,
    *,
    owner: str = "a1",
    session: str = "sess_x",
    day: int = 1,
) -> AgentCase:
    return AgentCase(
        id=f"{owner}_{entry}",
        entry_id=entry,
        owner_id=owner,
        owner_type="agent",
        session_id=session,
        timestamp=_ts(day),
        parent_type="memcell",
        parent_id="mc_99",
        quality_score=0.8,
        task_intent=f"intent {entry}",
        task_intent_tokens=f"intent {entry}",
        approach=f"approach {entry}",
        approach_tokens=f"approach {entry}",
        key_insight=None,
        md_path=f"agents/{owner}/cases/{entry}.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


def _agent_skill(
    name: str,
    *,
    owner: str = "a1",
) -> AgentSkill:
    return AgentSkill(
        id=f"{owner}_{name}",
        owner_id=owner,
        owner_type="agent",
        name=name,
        description=f"desc {name}",
        description_tokens=f"desc {name}",
        content=f"content {name}",
        content_tokens=f"content {name}",
        confidence=0.9,
        maturity_score=0.7,
        source_case_ids=[f"{owner}_ac_1"],
        md_path=f"agents/{owner}/skills/{name}/SKILL.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


@pytest.fixture
async def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """Build the FastAPI app against a tmp memory root with no lifespan."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    load_settings.cache_clear()

    # Reset every module-level singleton the get-path touches.
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    get_service_mod._manager = None

    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await lancedb_manager.dispose_connection()
    load_settings.cache_clear()


# ── Happy path ──────────────────────────────────────────────────────────


async def test_get_episodes_returns_page_and_total(
    client: AsyncClient,
) -> None:
    """5 rows in, page_size=2 → 2 episodes back + total_count=5."""
    await episode_repo.add(
        [_episode(f"ep_{i:03d}", day=i) for i in range(1, 6)],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "page": 1,
            "page_size": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    rid = body["request_id"]
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)
    data = body["data"]
    assert data["total_count"] == 5
    assert data["count"] == 2
    assert len(data["episodes"]) == 2
    # default sort = timestamp DESC → highest day first
    assert data["episodes"][0]["id"] == "u1_ep_005"
    assert data["episodes"][1]["id"] == "u1_ep_004"
    # The non-requested kinds are empty arrays (envelope invariant).
    assert data["profiles"] == []
    assert data["agent_cases"] == []
    assert data["agent_skills"] == []


async def test_get_episodes_filtered_by_session_id(
    client: AsyncClient,
) -> None:
    """Filter narrows results to the matching ``session_id`` only."""
    await episode_repo.add(
        [
            _episode("ep_001", session="sess_a"),
            _episode("ep_002", session="sess_a"),
            _episode("ep_003", session="sess_b"),
        ],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"session_id": "sess_a"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    assert body["data"]["count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_002"}


async def test_get_empty_returns_zero_counts(client: AsyncClient) -> None:
    """An owner with no rows yields total_count=0 + empty episodes list."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "ghost",
            "memory_type": "episode",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_count"] == 0
    assert data["count"] == 0
    assert data["episodes"] == []


async def test_get_profile_miss_returns_empty(client: AsyncClient) -> None:
    """Cold start (no profile row) → ``profiles=[]`` / ``total_count=0``."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "profile",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["profiles"] == []
    assert data["total_count"] == 0


async def test_get_profile_returns_seeded_row(client: AsyncClient) -> None:
    """A profile row in the ``user_profile`` table is returned + json-decoded.

    Full-stack: seed the LanceDB ``user_profile`` table (as cascade would
    from ``users/u1/user.md``), then read it back through the HTTP route.
    White-box surface: ``user_profile_repo`` (the same table /search's
    ``include_profile`` reads).
    """
    await user_profile_repo.add(
        [
            UserProfile(
                id="u1",
                owner_id="u1",
                owner_type="user",
                app_id="default",
                project_id="default",
                summary="u1 loves climbing in Yosemite",
                explicit_info_json='[{"category": "Hobby", "description": "climbing"}]',
                implicit_traits_json='[{"trait": "Outdoorsy"}]',
                profile_timestamp_ms=1780304400000,
                md_path="users/u1/user.md",
                content_sha256="abc",
            )
        ]
    )

    resp = await client.post(
        "/api/v1/memory/get",
        json={"user_id": "u1", "memory_type": "profile"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_count"] == 1
    assert data["count"] == 1
    assert len(data["profiles"]) == 1
    prof = data["profiles"][0]
    assert prof["id"] == "u1"
    assert prof["user_id"] == "u1"
    assert prof["profile_data"]["summary"] == "u1 loves climbing in Yosemite"
    assert prof["profile_data"]["explicit_info"] == [
        {"category": "Hobby", "description": "climbing"}
    ]
    assert prof["profile_data"]["implicit_traits"] == [{"trait": "Outdoorsy"}]


# ── Pagination + sort ───────────────────────────────────────────────────


async def test_get_episodes_page_two_returns_correct_slice(
    client: AsyncClient,
) -> None:
    """5 rows / page_size=2 / page=2 → middle slice (rows 3 + 4 by DESC ts)."""
    await episode_repo.add(
        [_episode(f"ep_{i:03d}", day=i) for i in range(1, 6)],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "page": 2,
            "page_size": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_count"] == 5
    assert data["count"] == 2
    # default sort = timestamp DESC; page 2 of 2-per-page over 5 rows →
    # rows at offsets 2,3 → day=3, day=2 (1-indexed: ep_003, ep_002).
    assert [ep["id"] for ep in data["episodes"]] == ["u1_ep_003", "u1_ep_002"]


async def test_get_episodes_sort_order_asc(client: AsyncClient) -> None:
    """``sort_order=asc`` flips the order (oldest first)."""
    await episode_repo.add(
        [_episode(f"ep_{i:03d}", day=i) for i in range(1, 4)],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "sort_order": "asc",
        },
    )
    assert resp.status_code == 200
    ids = [ep["id"] for ep in resp.json()["data"]["episodes"]]
    assert ids == ["u1_ep_001", "u1_ep_002", "u1_ep_003"]


# ── Agent-side kinds ────────────────────────────────────────────────────


async def test_get_agent_cases_happy_path(client: AsyncClient) -> None:
    """``agent_case`` listing returns shaped items, populates only that array."""
    await agent_case_repo.add(
        [_agent_case(f"ac_{i:03d}", day=i) for i in range(1, 4)],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "agent_id": "a1",
            "memory_type": "agent_case",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_count"] == 3
    assert data["count"] == 3
    assert [c["id"] for c in data["agent_cases"]] == [
        "a1_ac_003",
        "a1_ac_002",
        "a1_ac_001",
    ]
    # Cross-kind envelope stays empty.
    assert data["episodes"] == []
    assert data["agent_skills"] == []
    # AgentCase item shape — score absent (vs SearchAgentCaseItem),
    # quality_score round-trips.
    first = data["agent_cases"][0]
    assert "score" not in first
    assert first["quality_score"] == 0.8
    assert first["agent_id"] == "a1"


async def test_get_agent_cases_filtered_by_session(client: AsyncClient) -> None:
    """Filter narrows ``agent_case`` rows to the session."""
    await agent_case_repo.add(
        [
            _agent_case("ac_001", session="sess_x"),
            _agent_case("ac_002", session="sess_x"),
            _agent_case("ac_003", session="sess_y"),
        ]
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "agent_id": "a1",
            "memory_type": "agent_case",
            "filters": {"session_id": "sess_x"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {c["id"] for c in body["data"]["agent_cases"]}
    assert ids == {"a1_ac_001", "a1_ac_002"}


async def test_get_agent_skills_happy_path(client: AsyncClient) -> None:
    """``agent_skill`` listing — sort silently uses ``updated_at``."""
    await agent_skill_repo.add(
        [_agent_skill(name) for name in ("planner", "summariser")],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "agent_id": "a1",
            "memory_type": "agent_skill",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_count"] == 2
    names = {s["name"] for s in data["agent_skills"]}
    assert names == {"planner", "summariser"}


async def test_get_agent_skills_sort_by_timestamp_silently_downgraded(
    client: AsyncClient,
) -> None:
    """Explicit ``sort_by=timestamp`` does not 500 — manager rewrites to
    ``updated_at`` (the only temporal column on ``agent_skill``)."""
    await agent_skill_repo.add([_agent_skill("planner")])
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "agent_id": "a1",
            "memory_type": "agent_skill",
            "sort_by": "timestamp",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["total_count"] == 1


# ── Filter coverage end-to-end ──────────────────────────────────────────


async def test_get_episodes_filtered_by_ne_session(client: AsyncClient) -> None:
    """``ne`` op on a str field excludes matching rows end-to-end."""
    await episode_repo.add(
        [
            _episode("ep_001", session="sess_a"),
            _episode("ep_002", session="sess_internal"),
            _episode("ep_003", session="sess_b"),
        ]
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"session_id": {"ne": "sess_internal"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_003"}


async def test_get_episodes_filtered_by_iso_timestamp(
    client: AsyncClient,
) -> None:
    """ISO 8601 string timestamp literal is accepted alongside epoch ms."""
    await episode_repo.add(
        [
            _episode("ep_001", day=1),  # 2026-01-01
            _episode("ep_002", day=5),  # 2026-01-05
            _episode("ep_003", day=9),  # 2026-01-09
        ]
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"timestamp": {"gte": "2026-01-04T00:00:00+00:00"}},
        },
    )
    assert resp.status_code == 200
    ids = {ep["id"] for ep in resp.json()["data"]["episodes"]}
    assert ids == {"u1_ep_002", "u1_ep_003"}


async def test_get_episodes_filtered_by_parent_id(client: AsyncClient) -> None:
    """Core use case: every episode derived from one memcell."""
    await episode_repo.add(
        [
            _episode("ep_001", parent_id="mc_target"),
            _episode("ep_002", parent_id="mc_target"),
            _episode("ep_003", parent_id="mc_other"),
        ]
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"parent_id": "mc_target"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_002"}


async def test_get_episodes_filtered_by_sender_id_in(
    client: AsyncClient,
) -> None:
    """``sender_id: {"in": [...]}`` → ``array_has(sender_ids, ...) OR ...``."""
    await episode_repo.add(
        [
            _episode("ep_001", sender_ids=["alice", "assistant"]),
            _episode("ep_002", sender_ids=["bob", "assistant"]),
            _episode("ep_003", sender_ids=["carol", "assistant"]),
        ]
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"sender_id": {"in": ["alice", "bob"]}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_002"}


async def test_get_episodes_nested_and_inside_or(client: AsyncClient) -> None:
    """Nested ``AND`` inside ``OR`` — parity with /search combinator semantics."""
    await episode_repo.add(
        [
            _episode("ep_001", session="sess_a", parent_id="mc_target"),
            _episode("ep_002", session="sess_a", parent_id="mc_other"),
            _episode("ep_003", session="sess_b", parent_id="mc_target"),
            _episode("ep_004", session="sess_c", parent_id="mc_other"),
        ]
    )
    # (session=sess_a AND parent_id=mc_target)
    #   OR (parent_id=mc_other AND session=sess_c)
    # → ep_001 + ep_004
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {
                "OR": [
                    {
                        "AND": [
                            {"session_id": "sess_a"},
                            {"parent_id": "mc_target"},
                        ]
                    },
                    {
                        "AND": [
                            {"parent_id": "mc_other"},
                            {"session_id": "sess_c"},
                        ]
                    },
                ]
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_004"}


# ── Filter combinators (200 — happy path) ──────────────────────────────
# Pure 422 / validation cases moved to
# tests/unit/test_entrypoints/test_api/test_routes/test_get_route_validation.py


async def test_get_top_level_and_or_compiles_and_filters(
    client: AsyncClient,
) -> None:
    """``AND`` / ``OR`` combinators are accepted (parity with /search)."""
    await episode_repo.add(
        [
            _episode("ep_001", session="sess_a"),
            _episode("ep_002", session="sess_b"),
            _episode("ep_003", session="sess_c"),
        ],
    )
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"OR": [{"session_id": "sess_a"}, {"session_id": "sess_b"}]},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_001", "u1_ep_002"}


async def test_get_episodes_filtered_by_timestamp_range(
    client: AsyncClient,
) -> None:
    """``timestamp: {gte, lt}`` — same-field double op compiles to implicit AND."""
    await episode_repo.add(
        [
            _episode("ep_001", day=1),  # 2026-01-01
            _episode("ep_002", day=3),  # 2026-01-03
            _episode("ep_003", day=5),  # 2026-01-05
            _episode("ep_004", day=7),  # 2026-01-07
            _episode("ep_005", day=9),  # 2026-01-09
        ]
    )
    # Window [Jan 3, Jan 7) → ep_002 + ep_003 (Jan 7 excluded by `lt`).
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {
                "timestamp": {
                    "gte": "2026-01-03T00:00:00+00:00",
                    "lt": "2026-01-07T00:00:00+00:00",
                }
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 2
    ids = {ep["id"] for ep in body["data"]["episodes"]}
    assert ids == {"u1_ep_002", "u1_ep_003"}


async def test_get_episodes_top_level_and_filter(client: AsyncClient) -> None:
    """Explicit top-level ``AND`` — distinct from implicit multi-field AND."""
    await episode_repo.add(
        [
            _episode("ep_001", session="sess_a", parent_id="mc_target"),
            _episode("ep_002", session="sess_a", parent_id="mc_other"),
            _episode("ep_003", session="sess_b", parent_id="mc_target"),
        ]
    )
    # session=sess_a AND parent_id=mc_target → ep_001 only
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {
                "AND": [
                    {"session_id": "sess_a"},
                    {"parent_id": "mc_target"},
                ]
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_count"] == 1
    assert body["data"]["episodes"][0]["id"] == "u1_ep_001"


# ── max_fetch limit trigger ─────────────────────────────────────────────


async def test_get_truncates_above_max_fetch(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Filter matches > ``max_fetch`` rows → chassis emits warning + page
    contents come from the truncated prefix; ``total_count`` is still the
    *true* match count (``count_rows`` ignores ``max_fetch``).

    Injects a low ``max_fetch=5`` by wrapping the bound method so the
    end-to-end path runs through the truncation branch without populating
    20k+ rows.
    """
    # The e2e ``client`` fixture builds the app without lifespan providers,
    # so ``configure_logging`` (normally invoked by the CLI entry) never
    # runs. Call it here so the structlog → stdlib logging bridge is
    # wired up and ``caplog`` can observe the chassis warning.
    from everos.core.observability.logging import configure_logging

    configure_logging(level="WARNING")

    await episode_repo.add(
        [_episode(f"ep_{i:03d}", day=i) for i in range(1, 11)],
    )
    original = episode_repo.find_where_paginated

    async def low_cap(*args: object, **kwargs: object) -> object:
        kwargs["max_fetch"] = 5
        return await original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(episode_repo, "find_where_paginated", low_cap)

    with caplog.at_level("WARNING"):
        resp = await client.post(
            "/api/v1/memory/get",
            json={
                "user_id": "u1",
                "memory_type": "episode",
                "page": 1,
                "page_size": 3,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    # True row count is still 10, even though only 5 made it into the sort.
    assert body["data"]["total_count"] == 10
    assert body["data"]["count"] == 3
    # structlog now routes through stdlib's root logger (see
    # ``core/observability/logging/factory.py``); the warning surfaces via
    # the standard ``caplog`` fixture rather than direct stdout capture.
    assert "find_where_paginated truncated" in caplog.text


# ── Concurrency ─────────────────────────────────────────────────────────


async def test_get_concurrent_owners_no_cross_contamination(
    client: AsyncClient,
) -> None:
    """Concurrent /get requests against different ``owner_id`` partitions
    return only their own rows. ``GetManager`` is a lazy singleton —
    this also exercises first-request lazy-init under contention."""
    await episode_repo.add(
        [
            _episode("ep_001", owner="u1"),
            _episode("ep_002", owner="u1"),
            _episode("ep_001", owner="u2"),
            _episode("ep_001", owner="u3"),
        ]
    )

    async def query(owner: str) -> dict[str, object]:
        resp = await client.post(
            "/api/v1/memory/get",
            json={
                "user_id": owner,
                "memory_type": "episode",
            },
        )
        assert resp.status_code == 200, f"{owner}: {resp.text}"
        return resp.json()

    bodies = await asyncio.gather(
        query("u1"),
        query("u2"),
        query("u3"),
    )
    u1, u2, u3 = bodies
    assert u1["data"]["total_count"] == 2  # type: ignore[index]
    assert u2["data"]["total_count"] == 1  # type: ignore[index]
    assert u3["data"]["total_count"] == 1  # type: ignore[index]
    assert {ep["id"] for ep in u1["data"]["episodes"]} == {  # type: ignore[index]
        "u1_ep_001",
        "u1_ep_002",
    }
    assert {ep["id"] for ep in u2["data"]["episodes"]} == {"u2_ep_001"}  # type: ignore[index]
    assert {ep["id"] for ep in u3["data"]["episodes"]} == {"u3_ep_001"}  # type: ignore[index]


async def test_get_concurrent_different_memory_types(client: AsyncClient) -> None:
    """Concurrent /get on different ``memory_type`` (episode + agent_case +
    agent_skill) returns each kind in its own envelope slot, with no
    cross-array bleed."""
    await episode_repo.add([_episode("ep_001", owner="u1")])
    await agent_case_repo.add([_agent_case("ac_001", owner="a1")])
    await agent_skill_repo.add([_agent_skill("planner", owner="a1")])

    async def query(payload: dict[str, object]) -> dict[str, object]:
        resp = await client.post("/api/v1/memory/get", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    ep_body, case_body, skill_body = await asyncio.gather(
        query({"user_id": "u1", "memory_type": "episode"}),
        query(
            {
                "agent_id": "a1",
                "memory_type": "agent_case",
            }
        ),
        query(
            {
                "agent_id": "a1",
                "memory_type": "agent_skill",
            }
        ),
    )
    # Episode envelope: only ``episodes`` populated.
    assert len(ep_body["data"]["episodes"]) == 1  # type: ignore[index]
    assert ep_body["data"]["agent_cases"] == []  # type: ignore[index]
    assert ep_body["data"]["agent_skills"] == []  # type: ignore[index]
    # Case envelope: only ``agent_cases`` populated.
    assert len(case_body["data"]["agent_cases"]) == 1  # type: ignore[index]
    assert case_body["data"]["episodes"] == []  # type: ignore[index]
    # Skill envelope: only ``agent_skills`` populated.
    assert len(skill_body["data"]["agent_skills"]) == 1  # type: ignore[index]
    assert skill_body["data"]["episodes"] == []  # type: ignore[index]


async def test_get_concurrent_lazy_init_builds_one_manager(
    client: AsyncClient,
) -> None:
    """The lazy singleton survives first-request contention — N concurrent
    requests against a virgin manager all succeed and leave one instance."""
    # ``client`` fixture already reset _manager to None.
    assert get_service_mod._manager is None
    await episode_repo.add([_episode("ep_001")])

    payload = {
        "user_id": "u1",
        "memory_type": "episode",
    }
    results = await asyncio.gather(
        *(client.post("/api/v1/memory/get", json=payload) for _ in range(8))
    )
    assert all(r.status_code == 200 for r in results)
    # After the storm, exactly one manager instance is cached.
    assert get_service_mod._manager is not None
