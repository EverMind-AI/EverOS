"""Deleting a session detaches its episodes from their clusters, even on a retry.

A deletion that stops after rewriting the markdown leaves the episode in its cluster's
members; the retry finds no new entries to remove, yet must still finish that cleanup,
including when only another owner still has markdown for the session (#591). The
profile no longer reads cluster members (it folds episodes read back from markdown), so
what is pinned here is the membership itself.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from everalgo.types import ChatMessage

from everos.config import MemorizeSettings
from everos.core.errors import DeletionIncompleteError
from everos.infra.cascade._eraser import FencedIndexEraser
from everos.infra.cascade._fence import DeleteFence
from everos.infra.persistence.factories import build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.models import Cluster, Episode, MemCell, MemcellRecord
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, SessionKey, Stored
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
) -> None:
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


@pytest.mark.parametrize("session_only", [False, True])
@pytest.mark.parametrize("other_owner_remaining", [False, True])
async def test_session_retry_finishes_membership_cleanup(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    session_only: bool,
    other_owner_remaining: bool,
) -> None:
    await storage.create_schema()
    deps = make_deps(
        stores=build_stores(storage.root, storage.sqlite), llm=user_track_llm()
    )
    gone = await _seed(deps, "violet", _TS)
    kept = await _seed(deps, "cobalt", _TS + 1_000)
    await _cluster(deps, gone, kept)
    bob = replace(_SCOPE, owner_id="bob")
    if other_owner_remaining:
        await deps.stores.episode.append(
            bob,
            Episode(
                owner_id="bob",
                episode="Bob also joined violet",
                summary="violet",
                timestamp=_TS,
                session_id="violet",
                parent_id="mc_violet",
            ),
            only_if=NO_SOURCE_CHECK,
        )
    deleter = Deleter(
        stores=deps.stores,
        eraser=FencedIndexEraser(
            storage.repos,
            DeleteFence(),
            storage.sqlite.md_change_state,
        ),
        settings=MemorizeSettings(),
    )
    request = dict(
        app_id="default",
        project_id="default",
        agent_id=None,
        user_id=None if session_only else "alice",
        session_id="violet",
    )

    async def fail_cleanup(*args, **kwargs):
        raise OSError("interrupted member cleanup")

    with monkeypatch.context() as patch:
        patch.setattr(deleter, "_prune_clusters", fail_cleanup)
        with pytest.raises(DeletionIncompleteError, match="interrupted member cleanup"):
            await deleter.delete(**request)
    assert not await deps.stores.episode.exists(_SCOPE, [gone.entry_id])
    assert (
        gone.entry_id
        in (await deps.stores.cluster.list_for_owner(_SCOPE, "user_memory"))[0].members
    )
    if session_only:
        assert not await deps.stores.memcell.find_by_ids(["mc_violet"])
        # Recover Alice even when only Bob still has session markdown.
        owners = await deps.stores.owner_dir.owners_with_session(
            "default", "default", "violet"
        )
        assert _SCOPE not in owners
        assert (bob in owners) == other_owner_remaining

    await deleter.delete(**request)
    (cluster,) = await deps.stores.cluster.list_for_owner(_SCOPE, "user_memory")
    assert cluster.members == [kept.entry_id]
    assert cluster.count == 1 and cluster.last_ts == _TS + 1_000
