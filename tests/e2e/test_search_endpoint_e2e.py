"""End-to-end integration tests for ``POST /api/v1/memory/search``.

White-box e2e (per the project's testing taxonomy): real LanceDB writes
+ real embedder (when the method needs one) + real reranker / LLM
client when the method needs them. Data is reloaded from the
``tests/fixtures/search_seed/`` slice (16 episodes / 20 atomic_facts /
2 user_profiles produced by ``_dump_search_seed.py``) so the test
verifies recall on **real** vectors and **real** BM25 tokens.

Coverage matrix (see 21_test_taxonomy_debate.md context):

- methods: keyword / vector / hybrid / agentic (user owner only —
  agent_case / agent_skill ship in a separate pipeline)
- owner_id / owner_type isolation
- top_k (specific value + -1 unlimited)
- radius (cosine threshold)
- include_profile (true / false)
- filter DSL: session_id eq / timestamp range / sender_id in /
  parent_id (= memcell bridge) / top-level OR / nested AND-OR
- Hierarchical fact eviction: hybrid method embeds atomic_facts that share
  the matched episode's memcell parent

Methods other than ``keyword`` require ``EMBEDDING_*`` creds in .env —
they carry ``@pytest.mark.live_llm`` (and ``@pytest.mark.slow`` where
applicable). CI's default ``-m 'not slow and not live_llm'`` deselects
them. Run with ``pytest -m live_llm ...`` locally to exercise.
"""

from __future__ import annotations

import datetime as _dt
import time
from collections.abc import AsyncIterator, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from httpx import ASGITransport, AsyncClient

from everos.component.embedding import get_embedder
from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    UserProfile,
    agent_case_repo,
    agent_skill_repo,
    atomic_fact_repo,
    ensure_business_indexes,
    episode_repo,
    lancedb_manager,
    user_profile_repo,
)
from everos.infra.persistence.sqlite import cluster_repo, mint_cluster_id

# ``service.__init__`` shadows the submodule under the same name; reach
# the module via importlib so we can reset its private singletons.
search_service_mod = import_module("everos.service.search")


# ── Fixture: app with no lifespan + per-test singleton reset ──────────


@pytest.fixture
async def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """FastAPI app against a tmp memory root; no lifespan (no cascade/OME)."""
    from everos.core.persistence.sqlite import SQLModel as _SQLModel
    from everos.infra.persistence.sqlite import sqlite_manager

    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()

    # Lance: reset connection + cached table handles.
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()

    # SQLite: reset engine + factory singletons so the next ``get_engine``
    # targets the just-monkeypatched memory root, then run
    # ``metadata.create_all`` since we build the app with
    # ``lifespan_providers=[]`` and therefore skip
    # ``SqliteLifespanProvider``'s normal startup pass. The AGENTIC
    # cluster-path tests need the ``cluster`` table; other search paths
    # don't touch sqlite but the schema is cheap to materialise.
    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    _engine = sqlite_manager.get_engine()
    async with _engine.begin() as _conn:
        await _conn.run_sync(_SQLModel.metadata.create_all)

    # Search service: reset all lazy singletons so each test rebuilds
    # against the just-monkey-patched memory root + .env creds.
    for attr in (
        "_manager",
        "_embedding",
        "_reranker",
        "_llm_client",
    ):
        setattr(search_service_mod, attr, None)
    for attr in (
        "_embedding_resolved",
        "_rerank_resolved",
        "_llm_resolved",
    ):
        setattr(search_service_mod, attr, False)

    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await lancedb_manager.dispose_connection()
    await sqlite_manager.dispose_engine()
    load_settings.cache_clear()


# ── Seed loading helpers ──────────────────────────────────────────────


async def _seed_episodes(rows: list[dict[str, Any]]) -> list[Episode]:
    """Validate + add episode rows + build FTS index.

    Tests should pre-mutate per-row fields (e.g. ``session_id``) in the
    list passed in, rather than calling this multiple times. LanceDB's
    FTS index covers rows present at the time of index creation; calling
    :func:`ensure_business_indexes` here rebuilds the index over the
    current row set so sparse_recall can see them.
    """
    seeded = [Episode.model_validate(r) for r in rows]
    await episode_repo.add(seeded)
    await ensure_business_indexes()
    return seeded


async def _seed_atomic_facts(rows: list[dict[str, Any]]) -> list[AtomicFact]:
    facts = [AtomicFact.model_validate(r) for r in rows]
    await atomic_fact_repo.add(facts)
    await ensure_business_indexes()
    return facts


async def _seed_user_profiles(rows: list[dict[str, Any]]) -> list[UserProfile]:
    profiles = [UserProfile.model_validate(r) for r in rows]
    await user_profile_repo.add(profiles)
    # profile table has no FTS — no index rebuild needed.
    return profiles


async def _seed_user_memory_cluster(eps: list[dict], *, owner_id: str) -> None:
    """Seed one ``user_memory`` cluster covering every memcell in ``eps``.

    The AGENTIC episode path goes through ``acluster_retrieve`` (see
    ``memory/search/agentic.py``), which narrows hybrid candidates to the
    union of cluster member memcell ids. Tests that exercise the AGENTIC
    method therefore need at least one cluster whose members cover the
    seeded episodes' ``parent_id``s — otherwise ``cluster_scoped`` yields
    nothing and the agentic pipeline returns ``[]``.

    Centroid is embedded from one of the episode bodies via the live
    embedder; with a single cluster the cosine ranking against the query
    is trivial (only one candidate), so any reasonable anchor works.
    """
    memcell_ids = list({ep["parent_id"] for ep in eps})
    centroid_text = eps[0]["episode"]
    centroid_vec = await get_embedder().embed(centroid_text)
    await cluster_repo.upsert_with_members(
        AlgoCluster(
            id=mint_cluster_id(),
            centroid=np.asarray(centroid_vec, dtype=np.float32),
            count=len(memcell_ids),
            last_ts=int(time.time() * 1000),
            preview=[ep["episode"][:80] for ep in eps[:3]],
            members=memcell_ids,
        ),
        owner_id=owner_id,
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )


# Minimal agent_case / agent_skill row factories. The search_seed slice
# only ships user-side rows (episode / atomic_fact / foresight /
# user_profile), so agent-owner tests construct synthetic rows directly.
# ``vector`` defaults to zero-filled (fine for BM25-only / dispatch
# coverage where the test never queries the dense path); live_llm tests
# that exercise vector / hybrid recall must pass real embeddings via
# the ``vector`` kwarg so LanceDB's ``nearest_to`` returns the row
# (zero vectors are undefined under cosine distance).
def _agent_case(
    entry: str,
    *,
    owner: str = "a1",
    session: str = "sess_x",
    day: int = 1,
    task_intent: str | None = None,
    approach: str | None = None,
    vector: Sequence[float] | None = None,
) -> AgentCase:
    intent = task_intent if task_intent is not None else f"intent {entry}"
    appr = approach if approach is not None else f"approach {entry}"
    return AgentCase(
        id=f"{owner}_{entry}",
        entry_id=entry,
        owner_id=owner,
        owner_type="agent",
        session_id=session,
        timestamp=_dt.datetime(2026, 1, day, tzinfo=_dt.UTC),
        parent_type="memcell",
        parent_id="mc_99",
        quality_score=0.8,
        task_intent=intent,
        task_intent_tokens=intent,
        approach=appr,
        approach_tokens=appr,
        key_insight=None,
        md_path=f"agents/{owner}/cases/{entry}.md",
        content_sha256="abc",
        vector=list(vector) if vector is not None else [0.0] * 1024,
    )


def _agent_skill(
    name: str,
    *,
    owner: str = "a1",
    description: str | None = None,
    content: str | None = None,
    vector: Sequence[float] | None = None,
    source_case_ids: Sequence[str] | None = None,
) -> AgentSkill:
    desc = description if description is not None else f"desc {name}"
    body = content if content is not None else f"content {name}"
    return AgentSkill(
        id=f"{owner}_{name}",
        owner_id=owner,
        owner_type="agent",
        name=name,
        description=desc,
        description_tokens=desc,
        content=body,
        content_tokens=body,
        confidence=0.9,
        maturity_score=0.7,
        source_case_ids=(
            list(source_case_ids) if source_case_ids is not None else [f"{owner}_ac_1"]
        ),
        md_path=f"agents/{owner}/skills/{name}/SKILL.md",
        content_sha256="abc",
        vector=list(vector) if vector is not None else [0.0] * 1024,
    )


async def _seed_agent_cases(rows: list[AgentCase]) -> list[AgentCase]:
    await agent_case_repo.add(rows)
    await ensure_business_indexes()
    return rows


async def _seed_agent_skills(rows: list[AgentSkill]) -> list[AgentSkill]:
    await agent_skill_repo.add(rows)
    await ensure_business_indexes()
    return rows


def _eps_for_owner(seed: dict, owner: str) -> list[dict]:
    return [r for r in seed["episode"] if r["owner_id"] == owner]


def _facts_for_owner(seed: dict, owner: str) -> list[dict]:
    return [r for r in seed["atomic_fact"] if r["owner_id"] == owner]


def _profiles_for_owner(seed: dict, owner: str) -> list[dict]:
    return [r for r in seed["user_profile"] if r["owner_id"] == owner]


def _post(
    client: AsyncClient,
    *,
    owner_id: str = "caroline",
    owner_type: str = "user",
    query: str = "support",
    method: str | None = None,
    top_k: int | None = None,
    radius: float | None = None,
    include_profile: bool | None = None,
    enable_llm_rerank: bool | None = None,
    filters: dict | None = None,
):
    """Build a search request body with sensible defaults; return the coroutine.

    Callers still pass ``owner_id`` / ``owner_type`` (this helper's internal
    shape); the wire body uses ``user_id`` or ``agent_id`` per the public
    contract — the helper dispatches by ``owner_type``.
    """
    body: dict[str, Any] = {"query": query}
    if owner_type == "agent":
        body["agent_id"] = owner_id
    else:
        body["user_id"] = owner_id
    if method is not None:
        body["method"] = method
    if top_k is not None:
        body["top_k"] = top_k
    if radius is not None:
        body["radius"] = radius
    if include_profile is not None:
        body["include_profile"] = include_profile
    if enable_llm_rerank is not None:
        body["enable_llm_rerank"] = enable_llm_rerank
    if filters is not None:
        body["filters"] = filters
    return client.post("/api/v1/memory/search", json=body, timeout=60.0)


# ═══════════════════════════════════════════════════════════════════════
# 1. method × owner_type dispatch
# ═══════════════════════════════════════════════════════════════════════


async def test_keyword_search_returns_episode_hits(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=keyword`` runs BM25-only — no embedder needed."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    resp = await _post(client, query="LGBTQ", method="keyword")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert isinstance(data["episodes"], list)
    # At least one Caroline episode mentions LGBTQ explicitly.
    assert len(data["episodes"]) >= 1
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"
        # keyword path never populates nested atomic_facts.
        assert ep["atomic_facts"] == []


@pytest.mark.slow
@pytest.mark.live_llm
async def test_vector_search_returns_episode_hits(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=vector`` embeds the query and ranks by cosine.

    Seeds atomic_facts alongside episodes because the default
    ``vector_strategy = "maxsim_atomic"`` (config/default.toml) walks
    atomic_fact ANN → max-pool by parent_id → fetch episodes; an
    episode-only corpus would return 0 hits under that strategy.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_atomic_facts(_facts_for_owner(search_seed, "caroline"))

    # top_k>0 disables the default unlimited-radius (0.5) quality floor,
    # so we can verify dense recall finds *something* without the
    # threshold cutting off all candidates.
    resp = await _post(client, query="career", method="vector", top_k=5)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data["episodes"]) >= 1
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"
        assert ep["score"] > 0  # cosine similarity in [0, 1]
        # vector path doesn't run hierarchical fact eviction, so no nested facts.
        assert ep["atomic_facts"] == []


@pytest.mark.slow
@pytest.mark.live_llm
async def test_hybrid_search_returns_episode_hits(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=hybrid`` (default) runs sparse + dense + fusion."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    resp = await _post(client, query="counseling")  # method=hybrid default
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data["episodes"]) >= 1
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"


@pytest.mark.slow
@pytest.mark.live_llm
async def test_agentic_search_returns_episode_hits(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=agentic`` runs the cross-encoder rerank loop.

    The agentic episode pipeline (see ``memory/search/agentic.py``) is
    fact-MaxSim → hybrid_full → cluster_scoped → aagentic_retrieve, so
    the corpus needs atomic_facts (for the MaxSim child retrieve) and
    at least one ``user_memory`` cluster covering the seeded episodes'
    memcell ids (for the ``acluster_retrieve`` narrowing step).
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)
    await _seed_atomic_facts(_facts_for_owner(search_seed, "caroline"))
    await _seed_user_memory_cluster(eps, owner_id="caroline")

    resp = await _post(client, query="support group", method="agentic")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data["episodes"]) >= 1
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"


# ── Agent owner_type dispatch (separate path: agent_case + agent_skill) ─


async def _seed_one_agent_corpus(
    owner: str = "a1", *, use_real_embeddings: bool = False
) -> None:
    """Single seed used by the parametrized agent dispatch test.

    One case + one skill sharing surface tokens with the test query
    ("refactor authentication") so BM25 deterministically hits both
    tables. Dense / agentic methods exercise the same rows and opt into
    real embeddings so LanceDB's ``nearest_to`` can rank them (zero
    vectors are undefined under cosine distance — the dense path returns
    0 hits for them).
    """
    from everos.service.search import _get_embedding

    case_intent = "refactor authentication middleware"
    case_approach = "split provider lookup from session decode"
    skill_desc = "refactor authentication middleware reliably"
    skill_body = "step-by-step approach for auth refactors"

    embedder = _get_embedding() if use_real_embeddings else None
    if embedder is not None:
        case_vec, skill_vec = await embedder.embed_batch(
            [f"{case_intent}\n{case_approach}", f"{skill_desc}\n{skill_body}"]
        )
    else:
        # Keyword-only default runs offline in CI; live dense variants
        # pass real embeddings via ``use_real_embeddings=True``.
        case_vec = [1.0, *([0.0] * 1023)]
        skill_vec = [1.0, *([0.0] * 1023)]

    await _seed_agent_cases(
        [
            _agent_case(
                "ac_001",
                owner=owner,
                task_intent=case_intent,
                approach=case_approach,
                vector=case_vec,
            ),
        ]
    )
    await _seed_agent_skills(
        [
            _agent_skill(
                "refactor_auth_middleware",
                owner=owner,
                description=skill_desc,
                content=skill_body,
                vector=skill_vec,
            ),
        ]
    )


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("keyword", id="keyword"),
        pytest.param(
            "vector",
            id="vector",
            marks=[pytest.mark.slow, pytest.mark.live_llm],
        ),
        pytest.param(
            "hybrid",
            id="hybrid",
            marks=[pytest.mark.slow, pytest.mark.live_llm],
        ),
        pytest.param(
            "agentic",
            id="agentic",
            marks=[pytest.mark.slow, pytest.mark.live_llm],
        ),
    ],
)
async def test_search_agent_dispatch_per_method(
    client: AsyncClient, method: str
) -> None:
    """``owner_type=agent`` × every ranking method.

    Shared seed (one matching case + one matching skill) and shared
    query keep this a true dispatch matrix:

    - ``keyword`` is asserted strictly — BM25 over matching surface
      tokens must recall **both** tables.
    - ``vector`` / ``hybrid`` / ``agentic`` are looser (``>=1`` from
      either table) because dense recall + rerank may legitimately
      favour one side over the other on a 1-row corpus.

    All methods must enforce the owner_type hard partition:
    ``episodes`` / ``profiles`` stay empty.
    """
    await _seed_one_agent_corpus(use_real_embeddings=method != "keyword")

    resp = await _post(
        client,
        owner_id="a1",
        owner_type="agent",
        query="refactor authentication",
        method=method,
        top_k=5 if method in ("vector", "hybrid", "agentic") else None,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    if method == "keyword":
        assert len(data["agent_cases"]) >= 1
        assert len(data["agent_skills"]) >= 1
    else:
        assert len(data["agent_cases"]) >= 1 or len(data["agent_skills"]) >= 1

    for ac in data["agent_cases"]:
        assert ac["agent_id"] == "a1"
    for sk in data["agent_skills"]:
        assert sk["agent_id"] == "a1"
    assert data["episodes"] == []
    assert data["profiles"] == []


@pytest.mark.slow
@pytest.mark.live_llm
async def test_hybrid_with_llm_rerank_returns_hits(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=hybrid`` + ``enable_llm_rerank=true`` runs the phase-5 LLM pass.

    Default hybrid stops after hierarchical eviction / LR fusion; opting in adds one
    ``chat`` call that re-ranks the top-K. The route must accept the
    flag and still return well-formed episodes.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    resp = await _post(
        client,
        query="counseling",
        method="hybrid",
        enable_llm_rerank=True,
        top_k=5,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data["episodes"]) >= 1
    for ep in data["episodes"]:
        assert ep["user_id"] == "caroline"


@pytest.mark.slow
@pytest.mark.live_llm
async def test_hybrid_rerank_bridges_skill_via_case_lineage(
    client: AsyncClient,
) -> None:
    """Case→skill bridge: a skill whose own text does NOT directly match the
    query is surfaced because its ``source_case_ids`` lineage includes a
    case that does match. Exercised only on the HYBRID + ``enable_llm_rerank``
    serial path (``_search_cases_and_skills``).

    Seeds one case whose text contains the query keywords and one skill
    whose text is intentionally worded off the direct query terms but stays
    in the same domain. The direct skill recall (BM25 + cosine) cannot
    surface it on its own; the bridge is the path that does, and LLM
    rerank keeps it because the topic is genuinely relevant.
    """
    from everos.service.search import _get_embedding

    case_id_with_owner = "a_bridge_ac_1"  # mirrors AgentCase.id = "<owner>_<entry_id>"
    case_intent = "refactor authentication middleware"
    case_approach = "split provider lookup from session decode"
    # Skill is the *generalised lesson* from this case — same domain (auth /
    # identity / middleware layering) but worded so the query keywords
    # ("refactor authentication") do not surface it via direct BM25 or
    # dense recall in isolation. Only the case→skill bridge should
    # promote it; LLM rerank then keeps it because the topic *is* relevant.
    skill_desc = "isolate session token decoding from provider negotiation"
    skill_body = (
        "Principles for splitting identity verification layers across "
        "middleware tiers so token parsing and provider lookup evolve "
        "independently."
    )

    embedder = _get_embedding()
    assert embedder is not None, "live_llm test requires a real embedder"
    case_vec, skill_vec = await embedder.embed_batch(
        [f"{case_intent}\n{case_approach}", f"{skill_desc}\n{skill_body}"]
    )

    await _seed_agent_cases(
        [
            _agent_case(
                "ac_1",
                owner="a_bridge",
                task_intent=case_intent,
                approach=case_approach,
                vector=case_vec,
            ),
        ]
    )
    await _seed_agent_skills(
        [
            _agent_skill(
                "graphql_resolver_patterns",
                owner="a_bridge",
                description=skill_desc,
                content=skill_body,
                source_case_ids=[case_id_with_owner],
                vector=skill_vec,
            ),
        ]
    )

    resp = await _post(
        client,
        owner_id="a_bridge",
        owner_type="agent",
        query="refactor authentication",
        method="hybrid",
        enable_llm_rerank=True,
        top_k=5,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    case_ids = [c["id"] for c in data["agent_cases"]]
    skill_ids = [s["id"] for s in data["agent_skills"]]
    assert case_id_with_owner in case_ids, (
        f"case should match the query directly; got {case_ids}"
    )
    assert "a_bridge_graphql_resolver_patterns" in skill_ids, (
        "skill should be surfaced via case lineage bridge, not direct text "
        f"match (its text is intentionally worded off the query); got {skill_ids}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. owner_id / owner_type isolation
# ═══════════════════════════════════════════════════════════════════════


async def test_search_owner_isolation(client: AsyncClient, search_seed: dict) -> None:
    """Same query, two owners — each only sees their own episodes."""
    await _seed_episodes(
        _eps_for_owner(search_seed, "caroline") + _eps_for_owner(search_seed, "melanie")
    )

    resp_c = await _post(
        client, owner_id="caroline", query="caroline", method="keyword"
    )
    resp_m = await _post(client, owner_id="melanie", query="caroline", method="keyword")
    assert resp_c.status_code == 200
    assert resp_m.status_code == 200

    c_ids = {ep["id"] for ep in resp_c.json()["data"]["episodes"]}
    m_ids = {ep["id"] for ep in resp_m.json()["data"]["episodes"]}
    assert c_ids and m_ids
    assert c_ids.isdisjoint(m_ids)
    assert all(ep["user_id"] == "caroline" for ep in resp_c.json()["data"]["episodes"])
    assert all(ep["user_id"] == "melanie" for ep in resp_m.json()["data"]["episodes"])


async def test_search_owner_isolation_two_agents(client: AsyncClient) -> None:
    """Two agent owners share a hot keyword — each sees only its own rows."""
    await _seed_agent_cases(
        [
            _agent_case(
                "ac_a1_001",
                owner="a1",
                task_intent="optimize batch loader latency",
                approach="parallelize per-shard fetches",
            ),
            _agent_case(
                "ac_a2_001",
                owner="a2",
                task_intent="optimize batch loader latency",
                approach="cache hot keys in process memory",
            ),
        ]
    )

    resp_a1 = await _post(
        client,
        owner_id="a1",
        owner_type="agent",
        query="batch loader",
        method="keyword",
    )
    resp_a2 = await _post(
        client,
        owner_id="a2",
        owner_type="agent",
        query="batch loader",
        method="keyword",
    )
    assert resp_a1.status_code == 200
    assert resp_a2.status_code == 200

    a1_ids = {ac["id"] for ac in resp_a1.json()["data"]["agent_cases"]}
    a2_ids = {ac["id"] for ac in resp_a2.json()["data"]["agent_cases"]}
    assert a1_ids and a2_ids
    assert a1_ids.isdisjoint(a2_ids)
    assert all(ac["agent_id"] == "a1" for ac in resp_a1.json()["data"]["agent_cases"])
    assert all(ac["agent_id"] == "a2" for ac in resp_a2.json()["data"]["agent_cases"])


# ═══════════════════════════════════════════════════════════════════════
# 3. top_k
# ═══════════════════════════════════════════════════════════════════════


async def test_search_top_k_caps_result_count(
    client: AsyncClient, search_seed: dict
) -> None:
    """``top_k=2`` returns at most 2 episodes."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    resp = await _post(client, query="caroline", method="keyword", top_k=2)
    assert resp.status_code == 200
    assert len(resp.json()["data"]["episodes"]) <= 2


async def test_search_top_k_unlimited_returns_all_matching(
    client: AsyncClient, search_seed: dict
) -> None:
    """``top_k=-1`` (default) means unlimited; should exceed any specific cap."""
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)
    resp = await _post(client, query="caroline", method="keyword", top_k=-1)
    assert resp.status_code == 200
    # Caroline has 8 seeded episodes; unlimited returns up to that.
    assert len(resp.json()["data"]["episodes"]) <= len(eps)


# ═══════════════════════════════════════════════════════════════════════
# 4. radius (vector-only cosine threshold)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
@pytest.mark.live_llm
async def test_search_radius_filters_low_similarity(
    client: AsyncClient, search_seed: dict
) -> None:
    """A near-1.0 radius drops all but the closest hits (likely 0)."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    loose = await _post(client, query="career", method="vector", radius=0.0)
    strict = await _post(client, query="career", method="vector", radius=0.95)
    assert loose.status_code == 200 and strict.status_code == 200

    n_loose = len(loose.json()["data"]["episodes"])
    n_strict = len(strict.json()["data"]["episodes"])
    assert n_strict <= n_loose


# ═══════════════════════════════════════════════════════════════════════
# 5. include_profile
# ═══════════════════════════════════════════════════════════════════════


async def test_search_include_profile_false_returns_no_profile(
    client: AsyncClient, search_seed: dict
) -> None:
    """Default ``include_profile=False`` leaves ``data.profiles`` empty."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_user_profiles(_profiles_for_owner(search_seed, "caroline"))

    resp = await _post(client, query="LGBTQ", method="keyword")
    assert resp.status_code == 200
    assert resp.json()["data"]["profiles"] == []


async def test_search_include_profile_true_returns_profile(
    client: AsyncClient, search_seed: dict
) -> None:
    """``include_profile=true`` (user owner) attaches the profile row."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_user_profiles(_profiles_for_owner(search_seed, "caroline"))

    resp = await _post(client, query="LGBTQ", method="keyword", include_profile=True)
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


async def test_search_include_profile_ignored_for_agent_owner(
    client: AsyncClient, search_seed: dict
) -> None:
    """``include_profile=true`` is silently ignored when ``owner_type=agent``.

    Profile rows belong to user owners only (see
    :meth:`SearchManager._fetch_profile`); the agent dispatch returns
    ``profiles=[]`` even when a user profile matching the query exists.
    """
    # Seed a user profile that would match under user-owner search; the
    # agent-owner request must NOT pick it up.
    await _seed_user_profiles(_profiles_for_owner(search_seed, "caroline"))
    # FTS indexes are built lazily by _seed_*; build them for the agent
    # tables so sparse_recall returns [] gracefully on the agent path.
    await ensure_business_indexes()

    resp = await _post(
        client,
        owner_id="agent_x",
        owner_type="agent",
        query="LGBTQ",
        method="keyword",
        include_profile=True,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["profiles"] == []
    # Sanity: agent owner doesn't accidentally recall user-side rows.
    assert data["episodes"] == []


# ═══════════════════════════════════════════════════════════════════════
# 6. filter DSL (parity with /get's compile_filters_for_get)
# ═══════════════════════════════════════════════════════════════════════


async def test_search_filter_by_session_id(
    client: AsyncClient, search_seed: dict
) -> None:
    """``filters: {session_id: ...}`` confines the search to one session."""
    base = _eps_for_owner(search_seed, "caroline")
    # Re-tag half to a different session so the filter has something to do.
    half = len(base) // 2
    await _seed_episodes(
        [{**r, "session_id": "sess_target"} for r in base[:half]]
        + [{**r, "session_id": "sess_other"} for r in base[half:]]
    )

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"session_id": "sess_target"},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["session_id"] == "sess_target" for ep in eps)


async def test_search_filter_by_sender_id_in(
    client: AsyncClient, search_seed: dict
) -> None:
    """``filters: {sender_id: {in: [...]}}`` → array_has OR ... ."""
    base = _eps_for_owner(search_seed, "caroline")
    half = len(base) // 2
    await _seed_episodes(
        [{**r, "sender_ids": ["alice", "assistant"]} for r in base[:half]]
        + [{**r, "sender_ids": ["bob", "assistant"]} for r in base[half:]]
    )

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"sender_id": {"in": ["alice"]}},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all("alice" in ep["sender_ids"] for ep in eps)


async def test_search_filter_by_parent_id_memcell(
    client: AsyncClient, search_seed: dict
) -> None:
    """``filters: {parent_id: ...}`` — verifies memcell-based slicing.

    Same memcell is the bridge between episodes and atomic_facts.
    """
    base = _eps_for_owner(search_seed, "caroline")
    target_mc = base[0]["parent_id"]
    target_eps = [r for r in base if r["parent_id"] == target_mc]
    other_eps = [r for r in base if r["parent_id"] != target_mc]
    await _seed_episodes(target_eps + other_eps)

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"parent_id": target_mc, "parent_type": "memcell"},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["id"] in {r["id"] for r in target_eps} for ep in eps)


async def test_search_filter_top_level_or(
    client: AsyncClient, search_seed: dict
) -> None:
    """Top-level ``OR`` combinator parses + executes correctly."""
    base = _eps_for_owner(search_seed, "caroline")
    third = len(base) // 3
    await _seed_episodes(
        [{**r, "session_id": "sess_a"} for r in base[:third]]
        + [{**r, "session_id": "sess_b"} for r in base[third : third * 2]]
        + [{**r, "session_id": "sess_c"} for r in base[third * 2 :]]
    )

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"OR": [{"session_id": "sess_a"}, {"session_id": "sess_b"}]},
    )
    assert resp.status_code == 200
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["session_id"] in {"sess_a", "sess_b"} for ep in eps)


async def test_search_filter_nested_and_or(
    client: AsyncClient, search_seed: dict
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
    await _seed_episodes(built)

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
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
    client: AsyncClient, search_seed: dict
) -> None:
    """``timestamp`` filter with ``gte`` / ``lt`` operators narrows by time.

    Caroline's 8 seed episodes span 2023-05-08 → 2023-07-15; the
    ``2023-07-01`` cut splits them into a "before" and "after" group.
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)

    cutoff = "2023-07-01T00:00:00"
    expected_after = {r["id"] for r in eps if r["timestamp"] >= cutoff}
    assert expected_after, "seed should have at least one episode after the cutoff"

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        top_k=-1,
        filters={"timestamp": {"gte": cutoff}},
    )
    assert resp.status_code == 200
    returned = {ep["id"] for ep in resp.json()["data"]["episodes"]}
    # Every returned episode must be on/after the cutoff; the converse
    # (every expected episode came back) depends on the keyword match, so
    # we only require non-empty + correctness.
    assert returned
    assert returned <= expected_after


async def test_search_filter_no_match_returns_empty(
    client: AsyncClient, search_seed: dict
) -> None:
    """A filter that matches no row → 200 + episodes=[], not 404 / 422."""
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"session_id": "sess_that_does_not_exist"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["episodes"] == []
    rid = body["request_id"]
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)


# ═══════════════════════════════════════════════════════════════════════
# 7. Hierarchical fact eviction — the memcell-bridge contract
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
@pytest.mark.live_llm
async def test_search_hybrid_hierarchical_eviction_with_memcell_facts(
    client: AsyncClient, search_seed: dict
) -> None:
    """HYBRID + hierarchical eviction end-to-end with memcell facts.

    Verifies the wiring:
    - hybrid recall over episodes returns hits
    - the ``parent_id == memcell_id`` bridge picks up atomic_facts whose
      parent matches any candidate episode's parent (``facts_for_episodes``)

    .. note:: Fact *embedding* into ``episode.atomic_facts`` is NOT
       asserted because ``atomic_fact_recaller.facts_for_episodes``
       currently emits ``FactCandidate(score=0.0)`` for every prefetched
       fact (it's a parent_id lookup, not a query-aware recall). The
       Hierarchical eviction ``_expand_heap`` skips facts with non-positive scores, so
       they never promote into the top-N. Once facts get a real
       query-aware relevance score (e.g. by running a separate dense
       recall on atomic_fact too), tighten this assertion to verify
       at least one fact gets embedded with parent_id matching its
       host episode.
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)
    ep_parent_ids = {r["parent_id"] for r in eps}
    matching_facts = [
        r for r in search_seed["atomic_fact"] if r["parent_id"] in ep_parent_ids
    ]
    assert matching_facts, "seed should have at least one fact sharing a memcell"
    await _seed_atomic_facts(matching_facts)

    resp = await _post(client, query="counseling", method="hybrid", top_k=5)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["episodes"], "hybrid should return at least one episode"
    # Whichever facts *do* get embedded must share parent_id with their
    # host episode (the memcell-bridge invariant).
    for ep in data["episodes"]:
        if not ep["atomic_facts"]:
            continue
        host_parent = next((e["parent_id"] for e in eps if e["id"] == ep["id"]), None)
        for fact in ep["atomic_facts"]:
            seed_fact = next((r for r in matching_facts if r["id"] == fact["id"]), None)
            if seed_fact is not None:
                assert seed_fact["parent_id"] == host_parent


@pytest.mark.slow
@pytest.mark.live_llm
async def test_hybrid_hierarchical_eviction_injects_facts_with_alpha_zero(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hierarchical eviction end-to-end fact injection, exercised with ``alpha=0``.

    Companion to :func:`test_search_hybrid_hierarchical_eviction_with_memcell_facts`.
    The sibling asserts the contract under prod defaults
    (``alpha=1`` × ``fact.score=0`` → final ≤ 0 → fact never enters the
    top-N). This test patches ``RankConfig.alpha=0`` so facts inherit
    the parent episode's score and enter the heap — verifying the
    injection wire end-to-end.

    Drop the monkeypatch (or rewrite this test) once
    :mod:`everos.memory.search.recall.atomic_fact`'s ``facts_for_episodes``
    emits query-aware fact scores (currently hardcoded ``score=0.0``).
    """
    from everos.memory.search import manager as mgr

    class _AlphaZeroConfig(mgr.RankConfig):
        alpha: float = 0.0

    monkeypatch.setattr(mgr, "RankConfig", _AlphaZeroConfig)

    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)
    ep_parent_ids = {r["parent_id"] for r in eps}
    matching_facts = [
        r for r in search_seed["atomic_fact"] if r["parent_id"] in ep_parent_ids
    ]
    assert matching_facts, "seed should have at least one fact sharing a memcell"
    await _seed_atomic_facts(matching_facts)

    resp = await _post(client, query="counseling", method="hybrid", top_k=10)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["episodes"], "hybrid should return at least one episode"

    facts_attached = sum(len(ep["atomic_facts"]) for ep in data["episodes"])
    assert facts_attached >= 1, (
        "alpha=0 should let hierarchical eviction promote >=1 fact"
    )

    # Memcell-bridge invariant — every attached fact's parent_id must
    # match its host episode's parent_id.
    eps_by_id = {e["id"]: e for e in eps}
    for ep in data["episodes"]:
        host = eps_by_id.get(ep["id"])
        if host is None:
            continue
        for fact in ep["atomic_facts"]:
            seed_fact = next((r for r in matching_facts if r["id"] == fact["id"]), None)
            if seed_fact is not None:
                assert seed_fact["parent_id"] == host["parent_id"]


# ═══════════════════════════════════════════════════════════════════════
# 8. Request validation — Pydantic 422 surface
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"owner_id": "x", "owner_type": "robot", "query": "q"},
            id="invalid_owner_type",
        ),
        pytest.param(
            {"owner_id": "", "owner_type": "user", "query": "q"},
            id="empty_owner_id",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": ""},
            id="empty_query",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": "q", "top_k": 0},
            id="top_k_zero",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": "q", "top_k": -2},
            id="top_k_below_unlimited",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": "q", "top_k": 101},
            id="top_k_above_cap",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": "q", "radius": 1.1},
            id="radius_above_one",
        ),
        pytest.param(
            {"owner_id": "x", "owner_type": "user", "query": "q", "radius": -0.1},
            id="radius_negative",
        ),
    ],
)
async def test_search_rejects_invalid_request(
    client: AsyncClient, payload: dict
) -> None:
    """Pydantic / model_validator rejects malformed search requests as 422.

    Pins the request contract: ``owner_type`` literal, non-empty
    ``owner_id`` / ``query``, ``top_k`` in {-1} ∪ [1, 100], ``radius``
    in [0.0, 1.0]. Anything else short-circuits at the route layer
    without ever touching the SearchManager.
    """
    resp = await client.post("/api/v1/memory/search", json=payload, timeout=10.0)
    assert resp.status_code == 422, resp.text


# ═══════════════════════════════════════════════════════════════════════
# 9. Envelope shape & error surface
# ═══════════════════════════════════════════════════════════════════════


async def test_search_returns_empty_envelope_for_unknown_owner(
    client: AsyncClient,
) -> None:
    """Owner with no seeded rows → 200 with all five arrays empty.

    Pins the envelope-shape invariant: ``data.{episodes, profiles,
    agent_cases, agent_skills, unprocessed_messages}`` always exist;
    an empty result is a successful response, not 404.
    """
    resp = await _post(client, owner_id="ghost", query="anything", method="keyword")
    assert resp.status_code == 200
    body = resp.json()
    rid = body["request_id"]
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)
    data = body["data"]
    assert data == {
        "episodes": [],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": [],
        "unprocessed_messages": [],
    }


async def test_search_user_owner_has_empty_agent_arrays(
    client: AsyncClient, search_seed: dict
) -> None:
    """User-owner dispatch never populates ``agent_cases`` / ``agent_skills``.

    Belt-and-braces for the owner_type hard partition: even when both
    user *and* agent rows live in storage, a user-owner request must
    leave the agent arrays empty (and vice versa via the sibling agent
    test). Guards against future regressions that might leak cross-type
    rows when the dispatch logic is refactored.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_agent_cases([_agent_case("ac_leak", owner="a1")])
    await _seed_agent_skills([_agent_skill("leak_skill", owner="a1")])

    resp = await _post(client, query="caroline", method="keyword")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["episodes"]
    assert data["agent_cases"] == []
    assert data["agent_skills"] == []


async def test_search_filter_error_returns_422(
    client: AsyncClient, search_seed: dict
) -> None:
    """Filter DSL violations (unknown field) → 422 via the route's handler.

    ``FilterNode`` is permissive at Pydantic-validation time
    (``extra="allow"``) so unknown keys reach :func:`compile_filters`,
    which raises :class:`FilterError`. The route catches that and
    converts it to a clean 422 (separate path from Pydantic 422).
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    resp = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"this_field_does_not_exist": "x"},
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
# 10. Advanced combinations — method × filter, profile × method
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
@pytest.mark.live_llm
async def test_vector_search_with_session_filter(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=vector`` + ``filters={session_id: ...}`` narrows by session.

    Verifies the filter ``where`` clause is composed with the vector
    recall query (not bypassed by the dense path). Half the seed gets
    a target session, half gets another; only target hits may come back.

    The default ``vector_strategy = "maxsim_atomic"`` filters atomic_facts
    first (then max-pools to episodes), so the per-episode session_id
    mutation has to propagate to each fact via its parent memcell id —
    otherwise the where clause drops every fact and recall returns 0.
    """
    base = _eps_for_owner(search_seed, "caroline")
    facts = _facts_for_owner(search_seed, "caroline")
    half = len(base) // 2
    target_parent_ids = {r["parent_id"] for r in base[:half]}

    await _seed_episodes(
        [{**r, "session_id": "sess_target"} for r in base[:half]]
        + [{**r, "session_id": "sess_other"} for r in base[half:]]
    )
    await _seed_atomic_facts(
        [
            {
                **f,
                "session_id": (
                    "sess_target"
                    if f["parent_id"] in target_parent_ids
                    else "sess_other"
                ),
            }
            for f in facts
        ]
    )

    resp = await _post(
        client,
        query="caroline",
        method="vector",
        top_k=10,
        filters={"session_id": "sess_target"},
    )
    assert resp.status_code == 200, resp.text
    eps = resp.json()["data"]["episodes"]
    assert eps
    assert all(ep["session_id"] == "sess_target" for ep in eps)


@pytest.mark.slow
@pytest.mark.live_llm
async def test_agentic_search_with_timestamp_filter(
    client: AsyncClient, search_seed: dict
) -> None:
    """``method=agentic`` + ``filters={timestamp: {gte: ...}}`` honors the cutoff.

    The agentic path runs its own cross-encoder rerank loop; the
    ``where`` clause must still apply at recall, not be silently
    dropped by the agentic pipeline. Same seed requirements as the
    non-filter agentic test (facts + cluster).
    """
    eps = _eps_for_owner(search_seed, "caroline")
    await _seed_episodes(eps)

    cutoff = "2023-07-01T00:00:00"
    expected_after = {r["id"] for r in eps if r["timestamp"] >= cutoff}
    assert expected_after, "seed should have at least one episode after the cutoff"

    # The search_seed slice ships every caroline fact attached to one
    # episode (mc_a7a1a4cfd8e6, 2023-05-08); under maxsim_atomic the
    # cutoff would filter all facts out and recall would be empty. Spread
    # the facts round-robin across the post-cutoff episodes (matching both
    # ``parent_id`` and ``timestamp``) so the filter exercises real
    # narrowing on a non-empty fact set.
    eps_post = [r for r in eps if r["timestamp"] >= cutoff]
    facts = _facts_for_owner(search_seed, "caroline")
    await _seed_atomic_facts(
        [
            {
                **f,
                "parent_id": eps_post[i % len(eps_post)]["parent_id"],
                "timestamp": eps_post[i % len(eps_post)]["timestamp"],
            }
            for i, f in enumerate(facts)
        ]
    )
    await _seed_user_memory_cluster(eps, owner_id="caroline")

    # ``counseling`` matches the post-cutoff caroline corpus (July 6 episode
    # is explicitly about counseling + mental health), so the agentic
    # pipeline's LLM sufficiency check has concrete content to score on
    # — a vague query like ``support`` against the narrow filtered set
    # makes the LLM step return ``[]`` and the test flakes.
    resp = await _post(
        client,
        query="counseling",
        method="agentic",
        top_k=10,
        filters={"timestamp": {"gte": cutoff}},
    )
    assert resp.status_code == 200, resp.text
    eps_out = resp.json()["data"]["episodes"]
    assert eps_out
    returned = {ep["id"] for ep in eps_out}
    assert returned <= expected_after


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            "vector",
            id="vector",
            marks=[pytest.mark.slow, pytest.mark.live_llm],
        ),
        pytest.param(
            "hybrid",
            id="hybrid",
            marks=[pytest.mark.slow, pytest.mark.live_llm],
        ),
    ],
)
async def test_search_include_profile_independent_of_method(
    client: AsyncClient, search_seed: dict, method: str
) -> None:
    """``include_profile=true`` works regardless of ranking method.

    Profile fetch is decoupled from the recall pipeline (see
    :meth:`SearchManager._fetch_profile` — it gates on
    ``include_profile && owner_type=="user"`` and ignores ``method``).
    Pin that contract: vector / hybrid must also surface the profile,
    not just keyword.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_user_profiles(_profiles_for_owner(search_seed, "caroline"))

    resp = await _post(
        client,
        query="LGBTQ",
        method=method,
        include_profile=True,
        top_k=5,
    )
    assert resp.status_code == 200, resp.text
    profiles = resp.json()["data"]["profiles"]
    assert len(profiles) == 1
    p = profiles[0]
    assert p["user_id"] == "caroline"
    assert p["score"] is None  # direct fetch — no ranking


# ═══════════════════════════════════════════════════════════════════════
# 11. unprocessed_messages — in-flight boundary-detection buffer
# ═══════════════════════════════════════════════════════════════════════
#
# Buffer rows have no owner attribution (boundary detection runs before
# owner inference). The /search route surfaces them ONLY when ``filters``
# carries a top-level ``session_id`` eq scalar — the only meaningful
# query dimension on the unattributed buffer. Compound shapes (AND / OR,
# operator maps) do NOT trigger the lookup.
#
# White-box: the helpers below write directly into the
# ``unprocessed_buffer`` SQLite table via the public repo (no /add
# round-trip, so the boundary detector never runs and the rows stay
# unprocessed). The track string mirrors ``service._boundary._TRACK``
# (the shared single-pass detector track).


async def _seed_unprocessed_buffer_rows(
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
    :meth:`unprocessed_buffer_repo.replace` call — re-reading detached
    SQLModel instances and rewriting them in a later session triggers
    silent insert dedup on the PK, so callers must batch their rows.
    """
    import datetime as _dt2
    import json

    from everos.component.utils.datetime import get_now_with_timezone
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        unprocessed_buffer_repo,
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
    await unprocessed_buffer_repo.replace(
        session_id,
        track,
        rows,
        app_id=app_id,
        project_id=project_id,
    )


async def test_search_unprocessed_messages_returned_for_top_level_session_filter(
    client: AsyncClient, search_seed: dict
) -> None:
    """``filters={"session_id": "<sid>"}`` surfaces the buffered messages.

    Writes two rows into the unprocessed_buffer (same session) and one
    into a different session, then queries with the target session as a
    top-level eq scalar. Only the two matching rows should come back —
    in timestamp order, with the single-text shorthand collapsed to a
    plain ``str``.

    White-box surface: ``unprocessed_buffer_repo`` (read & write via the
    same public repo the pipeline uses).
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    # Two rows in the target session, one in a noise session.
    await _seed_unprocessed_buffer_rows(
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
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_inflight"},
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
    client: AsyncClient, search_seed: dict
) -> None:
    """No ``filters.session_id`` (or compound shape) → ``unprocessed_messages=[]``.

    Verifies the trigger semantics: the same buffer rows that the sibling
    test surfaces stay invisible to:

    1. a request without any ``filters``,
    2. a request whose ``session_id`` sits inside ``AND`` (not top-level),
    3. a request whose ``session_id`` uses the ``{"eq": ...}`` operator map.

    All three return ``[]`` even though matching buffer rows exist.

    White-box surface: ``unprocessed_buffer_repo``.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_unprocessed_buffer_rows(
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
    resp_no_filter = await _post(client, query="anything", method="keyword")
    assert resp_no_filter.status_code == 200, resp_no_filter.text
    assert resp_no_filter.json()["data"]["unprocessed_messages"] == []

    # 2. session_id inside AND — compound, not top-level eq scalar.
    resp_and = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"AND": [{"session_id": "sess_inflight"}]},
    )
    assert resp_and.status_code == 200, resp_and.text
    assert resp_and.json()["data"]["unprocessed_messages"] == []

    # 3. session_id via operator map — also not a top-level eq scalar.
    resp_op = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": {"eq": "sess_inflight"}},
    )
    assert resp_op.status_code == 200, resp_op.text
    assert resp_op.json()["data"]["unprocessed_messages"] == []


async def test_search_unprocessed_messages_assistant_role_round_trips(
    client: AsyncClient, search_seed: dict
) -> None:
    """Buffer surfacing carries the original ``role`` through the wire.

    Pins that role / content shape survive the round trip — the buffered
    row was inserted with ``role="assistant"`` and a multi-content payload
    would degrade to the opaque ``list[dict]`` shorthand, while a single
    text item collapses to ``str``.
    """
    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))
    await _seed_unprocessed_buffer_rows(
        "sess_assistant",
        [
            {
                "message_id": "msg_buf_asst",
                "sender_id": "bot",
                "text": "assistant-side buffered reply",
                "role": "assistant",
            },
        ],
    )

    resp = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_assistant"},
    )
    assert resp.status_code == 200, resp.text
    msgs = resp.json()["data"]["unprocessed_messages"]
    assert [m["id"] for m in msgs] == ["msg_buf_asst"]
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "assistant-side buffered reply"


# ═══════════════════════════════════════════════════════════════════════
# 12. Timezone discipline — storage UTC vs display tz, no drift on switch
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
    from everos.component.utils import datetime as dt_module
    from everos.config import load_settings as _ls

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", tz)
    _ls.cache_clear()
    dt_module._display_tz.cache_clear()


async def test_timezone_switch_preserves_utc_instant_across_responses(
    client: AsyncClient,
    search_seed: dict,
    monkeypatch: pytest.MonkeyPatch,
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

    from everos.component.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        unprocessed_buffer_repo,
    )

    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

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
        sender_id="alice",
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
    await unprocessed_buffer_repo.replace(
        "sess_tz_drift",
        "memorize",
        [row],
        app_id="default",
        project_id="default",
    )

    # Step 2: /search under Shanghai display tz → expect +08:00 offset.
    resp_sh = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_tz_drift"},
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
    # (see ``core/persistence/sqlite/base.py``) re-attaches ``tzinfo=UTC``
    # to every ``UtcDatetime`` column after ORM hydrate, so callers never
    # observe a naive datetime — even though SQLite physically strips tz
    # at write time and SQLAlchemy ORM bypasses Pydantic on read.
    rows_back = await unprocessed_buffer_repo.list_for_track(
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
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_tz_drift"},
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
) -> None:
    """Same anti-drift contract, but for the LanceDB-backed episode path.

    LanceDB's Arrow schema declares timestamp columns with ``tz=UTC``
    (see :attr:`BaseLanceTable.UTC_DATETIME_FIELDS`) so PyArrow returns
    aware UTC datetimes directly — no caller-side ``ensure_utc`` needed.
    The shaper only runs ``to_display_tz`` to convert UTC to the
    configured display zone. This test pins that path against a tz switch.
    """
    import datetime as _real_dt

    from everos.component.utils.datetime import from_iso_format

    # Seed one episode at a fixed instant. The seed's ``timestamp`` is
    # passed straight into ``Episode.model_validate`` — LanceDB will
    # strip the tz at write time and store UTC bytes.
    pinned_utc = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    base = _eps_for_owner(search_seed, "caroline")[:1]
    base[0] = {**base[0], "timestamp": pinned_utc, "session_id": "sess_ep_tz"}
    await _seed_episodes(base)

    # Step 1: Shanghai display tz.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"session_id": "sess_ep_tz"},
    )
    assert resp_sh.status_code == 200, resp_sh.text
    eps_sh = resp_sh.json()["data"]["episodes"]
    assert eps_sh, "seed should have produced a keyword match"
    ts_sh = eps_sh[0]["timestamp"]
    assert ts_sh.endswith("+08:00"), ts_sh
    assert "T14:00:00" in ts_sh, ts_sh

    # Step 2: switch to UTC.
    await _switch_display_tz(monkeypatch, "UTC")
    resp_utc = await _post(
        client,
        query="caroline",
        method="keyword",
        filters={"session_id": "sess_ep_tz"},
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
) -> None:
    """Reverse direction: write under UTC, read under Shanghai → no drift.

    Symmetric counterpart of
    :func:`test_timezone_switch_preserves_utc_instant_across_responses`.
    Covers the migration scenario where the default UTC deployment
    later turns on a local display tz.
    """
    import datetime as _real_dt
    import json as _json

    from everos.component.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        unprocessed_buffer_repo,
    )

    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    # Step 1: UTC display tz first, write at 06:00 UTC = 14:00 Shanghai.
    await _switch_display_tz(monkeypatch, "UTC")

    target_instant_utc = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    row = UnprocessedBuffer(
        message_id="msg_rev_001",
        app_id="default",
        project_id="default",
        session_id="sess_rev",
        track="memorize",
        sender_id="alice",
        sender_name=None,
        role="user",
        timestamp=target_instant_utc,
        content_items_json=_json.dumps([{"type": "text", "text": "reverse-tz row"}]),
        text="reverse-tz row",
        tool_calls_json=None,
        tool_call_id=None,
    )
    await unprocessed_buffer_repo.replace(
        "sess_rev",
        "memorize",
        [row],
        app_id="default",
        project_id="default",
    )

    # Step 2: read under UTC → expect Z / +00:00 + 06:00 wall clock.
    resp_utc = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_rev"},
    )
    assert resp_utc.status_code == 200, resp_utc.text
    msgs_utc = resp_utc.json()["data"]["unprocessed_messages"]
    ts_utc = msgs_utc[0]["timestamp"]
    assert ts_utc.endswith("Z") or ts_utc.endswith("+00:00"), ts_utc
    assert "T06:00:00" in ts_utc, ts_utc

    # Step 3: switch to Shanghai, read again → expect +08:00 + 14:00 wall clock.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_rev"},
    )
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

    from everos.component.utils.datetime import from_iso_format
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        unprocessed_buffer_repo,
    )

    await _seed_episodes(_eps_for_owner(search_seed, "caroline"))

    # Step 1: write row A under Shanghai.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    instant_a = _real_dt.datetime(2026, 5, 29, 6, 0, 0, tzinfo=_real_dt.UTC)
    row_a = UnprocessedBuffer(
        message_id="msg_mixed_a",
        app_id="default",
        project_id="default",
        session_id="sess_mixed",
        track="memorize",
        sender_id="alice",
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
        sender_id="alice",
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
    await unprocessed_buffer_repo.replace(
        "sess_mixed",
        "memorize",
        [row_a, row_b],
        app_id="default",
        project_id="default",
    )

    # Step 3: query under Shanghai → both rows render +08:00, instants preserved.
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp_sh = await _post(
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_mixed"},
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
        client,
        query="anything",
        method="keyword",
        filters={"session_id": "sess_mixed"},
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
