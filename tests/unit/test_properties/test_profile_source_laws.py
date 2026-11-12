"""Generated completed-deletion/index schedules over real episode files.

Current events must survive index lag. Indexed history remains best effort.
The oracle tracks live sources independently; deletion goes through Deleter,
including member cleanup, before selection starts. In-flight snapshots are
outside this contract. No LLM is involved.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import event, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.config import MemorizeSettings
from everos.infra.persistence.factories import build_stores
from everos.memory.events import ProfileClusterUpdated
from everos.memory.models import Cluster, Episode
from everos.memory.ports import NO_SOURCE_CHECK, Scope, Stored
from everos.memory.strategies.extract_user_profile import _select_via_cluster
from everos.service.memory._delete import Deleter
from tests.fakes import make_deps
from tests.fakes.storage import make_storage

_SCOPES = [
    Scope(app_id="default", project_id=project, owner_id="alice", owner_type="user")
    for project in ("one", "two")
]
_TS = 1_700_000_000_000


class _LaggingEraser:
    async def erase(self, scope, removed):
        pass  # Deliberately retain stale index rows after successful deletion.


class ProfileSourcesMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.tmp = tempfile.TemporaryDirectory(prefix="profile_sources_")
        self.runner = asyncio.Runner()
        self.storage = make_storage(Path(self.tmp.name))
        self.runner.run(self.storage.create_schema())
        self.deps = make_deps(
            stores=build_stores(self.storage.root, self.storage.sqlite)
        )
        self.store = self.deps.stores.episode
        self.deleter = Deleter(
            stores=self.deps.stores,
            eraser=_LaggingEraser(),
            settings=MemorizeSettings(),
        )
        self.rows: dict[tuple[int, int], Stored[Episode]] = {}
        self.live: set[tuple[int, int]] = set()
        self.indexed: set[tuple[int, int]] = set()
        self.last_ts = 0
        for space in range(2):
            self.append(space, 0, False)
            self.append(space, 1, False)

    @rule(space=st.integers(0, 1), slot=st.integers(0, 5), merged=st.booleans())
    def append(self, space: int, slot: int, merged: bool) -> None:
        key = space, slot
        if key in self.rows:
            return
        episode = Episode(
            owner_id="alice",
            episode=f"source {space}/{slot}",
            summary="source",
            timestamp=_TS + slot,
            parent_id=f"mc_{space}_{slot // 2}",
            parent_type="cluster" if merged else "memcell",
        )
        scope = _SCOPES[space]
        ref = self.runner.run(
            self.store.append(scope, episode, only_if=NO_SOURCE_CHECK)
        )
        assert ref is not None
        self.rows[key] = Stored(ref=ref, scope=scope, item=episode)
        self.live.add(key)
        self.runner.run(self._link(space, slot, ref.entry_id))
        event("append reflection" if merged else "append source")

    async def _link(self, space, slot, entry_id):
        clusters = self.deps.stores.cluster
        cid = f"cl_{space}_{slot % 2}"
        old = await clusters.list_for_owner(_SCOPES[space], "user_memory")
        members = next((c.members for c in old if c.id == cid), [])
        await clusters.upsert(
            _SCOPES[space],
            "user_memory",
            Cluster(
                id=cid,
                members=[*members, entry_id],
                centroid=(1.0,),
                count=len(members) + 1,
                last_ts=_TS + slot,
            ),
            member_type="episode",
        )

    @rule(space=st.integers(0, 1), slot=st.integers(0, 5))
    def delete(self, space: int, slot: int) -> None:
        key = space, slot
        if key not in self.live:
            return
        self.runner.run(
            self.deleter.delete_by_ids(
                app_id="default",
                project_id=_SCOPES[space].project_id,
                memory_type="episode",
                ids=[self.rows[key].ref.id],
                reason_code=None,
                note=None,
            )
        )
        self.live.remove(key)
        event("completed API deletion with retained memcell and lagging index")

    @rule(space=st.integers(0, 1), slot=st.integers(0, 5), visible=st.booleans())
    def project(self, space: int, slot: int, visible: bool) -> None:
        key = space, slot
        if key not in self.rows:
            return
        if visible and key in self.live:
            self.indexed.add(key)
        elif not visible:
            self.indexed.discard(key)
        self.deps.indexes.episode.rows = [self.rows[k] for k in sorted(self.indexed)]

    @rule(last_ts=st.integers(0, 6))
    def profile_time(self, last_ts: int) -> None:
        self.last_ts = _TS + last_ts

    @invariant()
    def select(self) -> None:
        for key, row in self.rows.items():
            if row.item.parent_type != "memcell":
                continue
            space, slot = key
            clusters = self.runner.run(
                self.deps.stores.cluster.list_for_owner(_SCOPES[space], "user_memory")
            )
            trigger = f"cl_{space}_{slot % 2}"
            selected = self.runner.run(
                _select_via_cluster(
                    ProfileClusterUpdated(
                        owner_id="alice",
                        project_id=_SCOPES[space].project_id,
                        cluster_id=trigger,
                        memcell_id=row.item.parent_id,
                        episode_entry_id=row.ref.entry_id,
                    ),
                    self.last_ts,
                    clusters,
                    self.deps,
                    _SCOPES[space],
                )
            )
            eligible = {
                r.ref.entry_id
                for k, r in self.rows.items()
                if k in self.live
                and k[0] == space
                and any(
                    c.id == f"cl_{space}_{k[1] % 2}"
                    and (c.id == trigger or c.last_ts > self.last_ts)
                    for c in clusters
                )
            }
            expected = {
                r.item.parent_id
                for k, r in self.rows.items()
                if k in self.live & self.indexed
                and k[0] == space
                and r.ref.entry_id in eligible
                and r.item.parent_type == "memcell"
            }
            if key in self.live:
                expected.add(row.item.parent_id)
            assert set(selected) == expected
            assert len(selected) == len(set(selected))

    def teardown(self) -> None:
        self.runner.run(self.storage.aclose())
        self.runner.close()
        self.tmp.cleanup()


TestProfileSources = ProfileSourcesMachine.TestCase
TestProfileSources.settings = settings(
    max_examples=80, stateful_step_count=30, deadline=None
)
