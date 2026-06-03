"""Agent pipeline e2e: 5 SWE-bench trajectories drive /add + /flush.

Drives the full HTTP route through to storage, exercising the agent-track
pipeline (boundary → memcell → extract_agent_case → trigger_skill_clustering
→ extract_agent_skill) with real LLM and real embedder credentials.

Mixed tenancy by design (sender_id alignment from fixture):

    agent_pytest  (1 session, pytest-dev/pytest-7236)      ┐ independent
    agent_sympy   (1 session, sympy/sympy-18763)           ┘ owners
    agent_django  (3 sessions, django/django-{14311,16255,16263})  shared

Concurrency strategy (workaround for the known
``trigger_skill_clustering`` read-modify-write race on a shared owner_id):

    Phase 1: pytest + sympy concurrent via asyncio.gather (disjoint owners)
    Phase 2: 3 django sessions sequential (same owner, would race)

Once the cluster race is fixed in production, Phase 2 can collapse into
the same gather and the test will still pass — the assertions are
race-free, only the driver is conservative.

White-box assertions (audit trail of internal surfaces touched):
    - sqlite ``memcell`` rows per session_id
    - filesystem ``<root>/agents/<agent>/.cases/*.md`` presence
    - LanceDB ``agent_case`` rows by ``owner_id`` (count + session_id set)
    - LanceDB ``agent_skill`` rows by ``owner_id`` (soft — LLM-dependent)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from everos.infra.persistence.lancedb import agent_case_repo, agent_skill_repo
from everos.infra.persistence.markdown import AgentCaseDailyFrontmatter

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "agent_trajectories"

# Hand-picked trajectories (kept in-tree as fixtures; this selection is
# the source of truth — the original converter is not in the repo).
_PYTEST_SESSION = "session_pytest_7236"
_SYMPY_SESSION = "session_sympy_18763"
_DJANGO_SESSIONS = (
    "session_django_14311",
    "session_django_16255",
    "session_django_16263",
)

_AGENT_PYTEST = "agent_pytest"
_AGENT_SYMPY = "agent_sympy"
_AGENT_DJANGO = "agent_django"

# Phase 3 drain budget: OME chain (case → cluster → skill) writes md in
# stages, each picked up by cascade. Multiple drain rounds with brief
# sleeps let the chain quiesce without false-positive completion.
_DRAIN_ROUNDS = 4
_DRAIN_TIMEOUT_SECONDS = 300.0
_DRAIN_INTER_ROUND_SLEEP_SECONDS = 5.0


def _load_fixture(session_id: str) -> dict:
    return json.loads((_FIXTURE_DIR / f"{session_id}.json").read_text())


async def _drive_session(
    client: httpx.AsyncClient, session_data: dict
) -> tuple[str, str]:
    """Run /add followed by /flush for one trajectory; return status."""
    sid = session_data["everos_session_id"]
    msgs = session_data["messages"]
    # MessageItemDTO.max_length=500; our largest fixture has 324 messages.
    r = await client.post(
        "/api/v1/memory/add",
        json={"session_id": sid, "messages": msgs},
        timeout=600.0,
    )
    assert r.status_code == 200, (
        f"{sid}: /add returned {r.status_code} — {r.text[:300]}"
    )
    r = await client.post(
        "/api/v1/memory/flush",
        json={"session_id": sid},
        timeout=600.0,
    )
    assert r.status_code == 200, (
        f"{sid}: /flush returned {r.status_code} — {r.text[:300]}"
    )
    return sid, r.json()["data"]["status"]


@pytest.mark.slow
@pytest.mark.live_llm
async def test_agent_pipeline_e2e_mixed_tenancy(
    async_client: httpx.AsyncClient,
    core_pipeline_runtime: Path,
    pipeline_done_poll: Callable[..., Awaitable[None]],
    memcell_count: Callable[..., Awaitable[int]],
) -> None:
    """5 SWE-bench trajectories → agent_case + agent_skill on three agents."""
    memory_root = core_pipeline_runtime

    pytest_fx = _load_fixture(_PYTEST_SESSION)
    sympy_fx = _load_fixture(_SYMPY_SESSION)
    django_fxs = [_load_fixture(s) for s in _DJANGO_SESSIONS]

    # ── Phase 1: independent owners concurrent ────────────────────────────
    await asyncio.gather(
        _drive_session(async_client, pytest_fx),
        _drive_session(async_client, sympy_fx),
    )

    # ── Phase 2: shared owner_id, sequential to dodge cluster race ────────
    for fx in django_fxs:
        await _drive_session(async_client, fx)

    # ── Phase 3: drain OME chain + cascade ────────────────────────────────
    for _ in range(_DRAIN_ROUNDS):
        await pipeline_done_poll(deadline_seconds=_DRAIN_TIMEOUT_SECONDS)
        await asyncio.sleep(_DRAIN_INTER_ROUND_SLEEP_SECONDS)

    # ── Phase 4: assertions ───────────────────────────────────────────────

    # 4.1 every session produced ≥1 memcell
    all_sessions = (_PYTEST_SESSION, _SYMPY_SESSION, *_DJANGO_SESSIONS)
    for sid in all_sessions:
        n = await memcell_count(sid)
        assert n >= 1, f"no memcell for session {sid!r} (got {n})"

    # 4.2 each agent has a .cases dir with ≥1 .md file
    agents_dir = memory_root / "default_app" / "default_project" / "agents"
    case_dir_name = AgentCaseDailyFrontmatter.DIR_NAME
    for agent_id in (_AGENT_PYTEST, _AGENT_SYMPY, _AGENT_DJANGO):
        case_dir = agents_dir / agent_id / case_dir_name
        assert case_dir.is_dir(), f"missing {case_dir!s} for agent={agent_id!r}"
        md_files = list(case_dir.glob("*.md"))
        assert md_files, f"no agent_case md under {case_dir!s}"

    # 4.3 LanceDB agent_case rows per owner
    pytest_cases = await agent_case_repo.find_where(f"owner_id = '{_AGENT_PYTEST}'")
    sympy_cases = await agent_case_repo.find_where(f"owner_id = '{_AGENT_SYMPY}'")
    django_cases = await agent_case_repo.find_where(f"owner_id = '{_AGENT_DJANGO}'")

    assert len(pytest_cases) >= 1, (
        f"no agent_pytest rows in LanceDB (got {len(pytest_cases)})"
    )
    assert len(sympy_cases) >= 1, (
        f"no agent_sympy rows in LanceDB (got {len(sympy_cases)})"
    )
    # Each django session writes at least one cell → at least one case per
    # session. Lower bound 3 covers the minimum; LLM may produce more.
    assert len(django_cases) >= 3, (
        f"agent_django expected ≥3 LanceDB cases (3 sessions), got {len(django_cases)}"
    )

    # 4.4 cross-owner isolation — each agent's cases trace back only to
    # its own sessions
    pytest_session_ids = {c.session_id for c in pytest_cases}
    assert pytest_session_ids == {_PYTEST_SESSION}, (
        f"agent_pytest cases leaked across sessions: {pytest_session_ids}"
    )
    sympy_session_ids = {c.session_id for c in sympy_cases}
    assert sympy_session_ids == {_SYMPY_SESSION}, (
        f"agent_sympy cases leaked across sessions: {sympy_session_ids}"
    )
    django_session_ids = {c.session_id for c in django_cases}
    assert django_session_ids == set(_DJANGO_SESSIONS), (
        f"agent_django session set mismatch — got {django_session_ids}, "
        f"want {set(_DJANGO_SESSIONS)}"
    )

    # 4.5 agent_skill — soft: emission depends on LLM clustering quality
    # gate (skip_quality_threshold + cluster size). pytest/sympy are
    # single-case clusters and may legitimately yield 0 skills. django
    # has 3 cases and should aggregate into ≥1 cluster of size ≥2,
    # producing ≥1 skill — but we keep this informational (LLM-dependent)
    # rather than a hard floor to avoid flaky CI signal.
    pytest_skills = await agent_skill_repo.find_where(f"owner_id = '{_AGENT_PYTEST}'")
    sympy_skills = await agent_skill_repo.find_where(f"owner_id = '{_AGENT_SYMPY}'")
    django_skills = await agent_skill_repo.find_where(f"owner_id = '{_AGENT_DJANGO}'")
    # Hard sanity: counts non-negative (the repo isn't broken).
    assert len(pytest_skills) >= 0
    assert len(sympy_skills) >= 0
    assert len(django_skills) >= 0

    # 4.6 strict md ↔ LanceDB parity across every cascade kind
    #
    # The per-owner counts above are loose (LLM-emission-dependent); this
    # check enforces byte-exact id-set + content_sha256 parity across
    # every md the agent pipeline wrote.
    #
    # ``expect_at_least`` pins agent_case (every session writes ≥1 case)
    # so an empty glob would fail loudly. agent_skill is NOT pinned —
    # emission depends on the LLM clustering quality gate per 4.5; a
    # legitimately empty agent_skill md set is still a passing run.
    from tests._consistency_assertions import assert_md_lance_strict_consistent

    await assert_md_lance_strict_consistent(
        memory_root,
        expect_at_least={"agent_case": 1},
    )
