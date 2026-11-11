"""Races of a deletion with what runs beside it, each pinned on its own.

``tests/unit/test_properties/test_memory_delete_jitter.py`` found them by
chance (about one run in fifteen under load); these replay each one in the
order that broke it:

- a client that goes away while ``/memory/delete`` reads the session's group
  binding must not cut that read: it runs inside the shielded deletion, so a
  cancelled request never interrupts a database call half way (the cut call
  closed its connection under the rollback that followed: "no active
  connection");
- a deletion by id must not leave its episode in a cluster when a clustering
  run, which found the episode still in md before the deletion removed it,
  writes it into a cluster while the deletion looks for its clusters;
- (a third, a skill read meeting a ``SKILL.md`` a concurrent deletion of the
  agent moved away between its check and its read, has no deterministic
  replay: the window is inside one reader call.)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from everos.config import MemorizeSettings
from everos.core.errors import InvalidInputError
from everos.infra.persistence.factories import build_stores
from everos.memory import user_cluster_lock
from everos.memory.deps import Stores
from everos.memory.models import Cluster, Episode
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, SessionKey
from everos.service.memory._delete import Deleter
from tests.fakes import InMemoryGroupStore, make_memory_service, make_stores
from tests.fakes.storage import Storage

_U1 = Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")
_TS = 1_747_216_800_000


class _SlowBinding(InMemoryGroupStore):
    """A group store whose binding read waits until the test lets it finish."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False

    async def binding(self, session: SessionKey) -> str | None:
        self.entered.set()
        await self.release.wait()
        self.finished = True
        return "trip"


async def test_a_client_going_away_does_not_cut_the_binding_read() -> None:
    groups = _SlowBinding()
    service = make_memory_service(stores=make_stores(groups=groups))
    request = asyncio.ensure_future(
        service.delete(
            app_id="a", project_id="p", user_id=None, agent_id=None, session_id="s1"
        )
    )
    await groups.entered.wait()

    request.cancel()  # the client went away mid-read
    with pytest.raises(asyncio.CancelledError):
        await request
    groups.release.set()
    await service.aclose()  # waits for what still runs

    assert groups.finished  # the read ran to its end, inside the shield


async def test_a_bound_session_is_still_refused() -> None:
    groups = _SlowBinding()
    groups.release.set()
    service = make_memory_service(stores=make_stores(groups=groups))
    with pytest.raises(InvalidInputError, match="belongs to a group"):
        await service.delete(
            app_id="a", project_id="p", user_id=None, agent_id=None, session_id="s1"
        )


class _Eraser:
    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        return None

    async def erase_owner(self, scope: Scope, kinds: Any) -> None:
        return None


@pytest.fixture
async def stores(storage: Storage) -> AsyncIterator[Stores]:
    await storage.create_schema()
    yield build_stores(storage.root, storage.sqlite)


async def test_a_clustering_run_in_flight_cannot_keep_a_deleted_episode(
    stores: Stores,
) -> None:
    (episode,) = await stores.episode.append_many(
        _U1,
        [
            Episode(
                owner_id="u1",
                episode="e",
                summary="s",
                subject="s",
                timestamp=_TS,
                session_id="s1",
                parent_id="mc_1",
            )
        ],
        only_if=NO_SOURCE_CHECK,
    )
    checked = asyncio.Event()
    looked = asyncio.Event()

    async def clustering() -> None:
        # What trigger_profile_clustering does: under the clustering lock it
        # finds the episode still in md, then writes it into a cluster -- here
        # only once the deletion looked for the episode's clusters (or could
        # not get to that before this run let the lock go).
        async with user_cluster_lock("a", "p", "u1"):
            assert await stores.episode.exists(_U1, [episode.entry_id])
            checked.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(looked.wait(), 0.3)
            await stores.cluster.upsert(
                _U1,
                "user_memory",
                Cluster(
                    id="cl_race",
                    centroid=(1.0,),
                    last_ts=_TS,
                    members=[episode.entry_id],
                ),
                member_type="episode",
            )

    class _Clusters:
        """The cluster store, telling the clustering run once it was asked."""

        def __getattr__(self, name: str) -> Any:
            return getattr(stores.cluster, name)

        async def find_cluster_for_member(self, *args: Any) -> str | None:
            looked.set()
            return await stores.cluster.find_cluster_for_member(*args)

    run = asyncio.ensure_future(clustering())
    await checked.wait()
    deleter = Deleter(
        stores=dataclasses.replace(stores, cluster=_Clusters()),
        eraser=_Eraser(),  # type: ignore[arg-type]
        settings=MemorizeSettings(),
    )
    await deleter.delete_by_ids(
        app_id="a",
        project_id="p",
        memory_type="episode",
        ids=[episode.id],
        reason_code=None,
        note=None,
    )
    await run

    assert (
        await stores.cluster.find_cluster_for_member(_U1, "episode", episode.entry_id)
        is None
    )
