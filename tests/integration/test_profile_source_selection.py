"""Profile selection over real markdown/SQLite with controlled index lag."""

from __future__ import annotations

import pytest
from everalgo.types import ChatMessage

from everos.config import MemorizeSettings
from everos.infra.cascade._eraser import FencedIndexEraser
from everos.infra.cascade._fence import DeleteFence
from everos.infra.persistence.factories import build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.events import ProfileClusterUpdated
from everos.memory.models import Cluster, Episode, MemCell, MemcellRecord, Profile
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, SessionKey, Stored
from everos.memory.strategies import extract_user_profile
from everos.service.memory._delete import Deleter
from tests.fakes import make_deps, user_track_llm
from tests.fakes.storage import Storage

_SCOPE = Scope(
    app_id="default", project_id="default", owner_id="alice", owner_type="user"
)
_TS = 1_700_000_000_000


async def _seed(deps: MemoryDeps, name: str, timestamp: int) -> MemoryRef:
    memcell_id = f"mc_{name}"
    await deps.stores.memcell.insert_many(
        SessionKey(app_id="default", project_id="default", session_id=name),
        [
            MemcellRecord(
                memcell_id=memcell_id,
                cell=MemCell(
                    items=[
                        ChatMessage(
                            id=f"{name}_message",
                            role="user",
                            sender_id="alice",
                            content=f"My unique preference is {name}.",
                            timestamp=timestamp,
                        )
                    ],
                    timestamp=timestamp,
                ),
                raw_type="conversation",
                message_ids=[f"{name}_message"],
                sender_ids=["alice"],
            )
        ],
    )
    episode = Episode(
        owner_id="alice",
        episode=f"Alice prefers {name}.",
        summary=name,
        timestamp=timestamp,
        session_id=name,
        parent_id=memcell_id,
    )
    ref = await deps.stores.episode.append(_SCOPE, episode, only_if=NO_SOURCE_CHECK)
    assert ref is not None
    deps.indexes.episode.seed(Stored(ref=ref, scope=_SCOPE, item=episode))
    return ref


async def _cluster(
    deps: MemoryDeps, *refs: MemoryRef, last_ts: int = _TS + 1_000
) -> ProfileClusterUpdated:
    await deps.stores.cluster.upsert(
        _SCOPE,
        "user_memory",
        Cluster(
            id="cl_shared",
            members=[r.entry_id for r in refs],
            centroid=(1.0,),
            count=len(refs),
            last_ts=last_ts,
        ),
        member_type="episode",
    )
    return ProfileClusterUpdated(
        owner_id="alice", cluster_id="cl_shared", memcell_id="mc_violet"
    )


def _prompt(deps: MemoryDeps) -> str:
    return "\n".join(m.content for call in deps.llm.calls for m in call.messages)


@pytest.mark.parametrize("indexed", [False, True])
async def test_first_profile_uses_durable_episode_before_indexing(
    storage: Storage, indexed: bool
) -> None:
    await storage.create_schema()
    deps = make_deps(
        stores=build_stores(storage.root, storage.sqlite), llm=user_track_llm()
    )
    ref = await _seed(deps, "violet", _TS)
    event = await _cluster(deps, ref)
    if not indexed:
        deps.indexes.episode.rows.clear()

    await extract_user_profile(event, deps)

    profile = await deps.stores.profile.read(_SCOPE)
    assert profile is not None
    assert profile.explicit_info
    assert "My unique preference is violet." in _prompt(deps)


@pytest.mark.parametrize("deletion", ["by_ids", "owner_session"])
@pytest.mark.parametrize("stale_index", [False, True])
async def test_queued_profile_event_does_not_reselect_deleted_source(
    storage: Storage, deletion: str, stale_index: bool
) -> None:
    await storage.create_schema()
    deps = make_deps(
        stores=build_stores(storage.root, storage.sqlite), llm=user_track_llm()
    )
    gone = await _seed(deps, "violet", _TS)
    kept = await _seed(deps, "cobalt", _TS + 1_000)
    event = await _cluster(deps, gone, kept)
    deleter = Deleter(
        stores=deps.stores,
        eraser=FencedIndexEraser(
            storage.repos, DeleteFence(), storage.sqlite.md_change_state
        ),
        settings=MemorizeSettings(),
    )
    if deletion == "by_ids":
        assert (
            await deleter.delete_by_ids(
                app_id="default",
                project_id="default",
                memory_type="episode",
                ids=[gone.id],
                reason_code=None,
                note=None,
            )
            == 1
        )
    else:
        await deleter.delete(
            app_id="default",
            project_id="default",
            user_id="alice",
            agent_id=None,
            session_id="violet",
        )
    assert await deps.stores.episode.exists(_SCOPE, [gone.entry_id, kept.entry_id]) == {
        kept.entry_id
    }
    assert await deps.stores.memcell.find_by_ids(["mc_violet"])
    assert await deps.stores.cluster.list_for_owner(_SCOPE, "user_memory")
    if not stale_index:
        deps.indexes.episode.rows = [
            row for row in deps.indexes.episode.rows if row.ref.id != gone.id
        ]

    await extract_user_profile(event, deps)

    assert await deps.stores.profile.read(_SCOPE) is not None
    prompt = _prompt(deps)
    assert "My unique preference is cobalt." in prompt
    assert "My unique preference is violet." not in prompt


async def test_current_and_fresh_clusters_keep_unique_source_memcells(
    storage: Storage,
) -> None:
    await storage.create_schema()
    deps = make_deps(
        stores=build_stores(storage.root, storage.sqlite), llm=user_track_llm()
    )
    current = await _seed(deps, "violet", _TS)
    fresh = await _seed(deps, "cobalt", _TS + 1_000)
    stale = await _seed(deps, "amber", _TS - 1_000)
    duplicate, merged = await deps.stores.episode.append_many(
        _SCOPE,
        [
            Episode(
                owner_id="alice",
                episode="Another view of the same conversation.",
                summary="duplicate source",
                timestamp=_TS,
                parent_id="mc_violet",
            ),
            Episode(
                owner_id="alice",
                episode="A reflection with no source memcell.",
                summary="reflection",
                timestamp=_TS + 1_000,
                parent_id="cl_reflected",
                parent_type="cluster",
            ),
        ],
        only_if=NO_SOURCE_CHECK,
    )
    event = await _cluster(deps, current, duplicate, last_ts=_TS)
    for name, refs, timestamp in (
        ("cl_fresh", [fresh, merged], _TS + 1_000),
        ("cl_stale", [stale], _TS - 1_000),
    ):
        await deps.stores.cluster.upsert(
            _SCOPE,
            "user_memory",
            Cluster(
                id=name,
                members=[r.entry_id for r in refs],
                centroid=(1.0,),
                count=len(refs),
                last_ts=timestamp,
            ),
            member_type="episode",
        )
    # The current cluster is older than the profile but still qualifies by id.
    await deps.stores.profile.write(
        _SCOPE, Profile(owner_id="alice", summary="prior", timestamp=_TS + 500)
    )
    deps.indexes.episode.rows.clear()

    await extract_user_profile(event, deps)

    prompt = _prompt(deps)
    assert prompt.count("My unique preference is violet.") == 1
    assert prompt.count("My unique preference is cobalt.") == 1
    assert "My unique preference is amber." not in prompt
    profile = await deps.stores.profile.read(_SCOPE)
    assert profile is not None and profile.timestamp == _TS + 1_000


async def test_removed_cluster_does_not_replay_its_event(storage: Storage) -> None:
    await storage.create_schema()
    deps = make_deps(
        stores=build_stores(storage.root, storage.sqlite), llm=user_track_llm()
    )
    ref = await _seed(deps, "violet", _TS)
    event = await _cluster(deps, ref)
    await deps.stores.cluster.delete(event.cluster_id)

    await extract_user_profile(event, deps)

    assert await deps.stores.profile.read(_SCOPE) is None
    assert deps.llm.calls == []
