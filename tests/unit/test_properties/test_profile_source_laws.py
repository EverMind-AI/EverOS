"""Generated selection/deletion schedules over the real episode files.

The oracle is a set of live source conversations, independent of index
visibility. Cluster snapshots may predate deletion, but selection must
include all surviving eligible sources exactly once and no removed,
reflection, or foreign-project source. No LLM is involved in these laws.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import event, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown.stores import MarkdownEpisodeStore
from everos.memory.events import ProfileClusterUpdated
from everos.memory.models import Cluster, Episode
from everos.memory.ports import NO_SOURCE_CHECK, Scope, Stored
from everos.memory.strategies.extract_user_profile import _select_via_cluster
from tests.fakes import make_deps, make_stores

_SCOPES = [
    Scope(app_id="default", project_id=project, owner_id="alice", owner_type="user")
    for project in ("one", "two")
]


class ProfileSourcesMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.tmp = tempfile.TemporaryDirectory(prefix="profile_sources_")
        self.runner = asyncio.Runner()
        self.store = MarkdownEpisodeStore(MemoryRoot(Path(self.tmp.name)))
        self.deps = make_deps(stores=make_stores(episode=self.store))
        self.rows: dict[tuple[int, int], Stored[Episode]] = {}
        self.live: set[tuple[int, int]] = set()
        self.members: set[tuple[int, int]] = set()
        self.pending: tuple[int, int, int, list[Cluster]] | None = None
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
            timestamp=1_700_000_000_000 + slot,
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
        self.members.add(key)
        event("append reflection" if merged else "append source")

    @rule(space=st.integers(0, 1), slot=st.integers(0, 5), detach=st.booleans())
    def delete(self, space: int, slot: int, detach: bool) -> None:
        key = space, slot
        if key not in self.live:
            return
        self.runner.run(
            self.store.remove(_SCOPES[space], [self.rows[key].ref.entry_id])
        )
        self.live.remove(key)
        if detach:
            self.members.discard(key)
        event("delete after snapshot" if self.pending else "delete before snapshot")

    @rule(visible=st.booleans(), stale=st.booleans())
    def project(self, visible: bool, stale: bool) -> None:
        keys = self.rows.keys() if stale else self.live
        self.deps.indexes.episode.rows = [self.rows[k] for k in keys] if visible else []
        event("stale index" if stale and visible else "fresh or empty index")

    @rule(space=st.integers(0, 1), trigger=st.integers(0, 1), last_ts=st.integers(0, 3))
    def snapshot(self, space: int, trigger: int, last_ts: int) -> None:
        clusters = []
        for group in range(2):
            keys = sorted(
                k for k in self.members if k[0] == space and k[1] % 2 == group
            )
            if keys:
                clusters.append(
                    Cluster(
                        id=f"cl_{group}",
                        members=[self.rows[k].ref.entry_id for k in keys],
                        centroid=(1.0,),
                        count=len(keys),
                        last_ts=group + 1,
                    )
                )
        self.pending = space, trigger, last_ts, clusters
        event("snapshot")

    @invariant()
    def select(self) -> None:
        if self.pending is None:
            return
        space, trigger, last_ts, clusters = self.pending
        scope = _SCOPES[space]
        selected = self.runner.run(
            _select_via_cluster(
                ProfileClusterUpdated(
                    owner_id="alice",
                    project_id=scope.project_id,
                    cluster_id=f"cl_{trigger}",
                    memcell_id=f"mc_{space}_0",
                ),
                last_ts,
                clusters,
                self.deps,
                scope,
            )
        )
        eligible = {
            entry
            for cluster in clusters
            if cluster.id == f"cl_{trigger}" or cluster.last_ts > last_ts
            for entry in cluster.members
        }
        expected = {
            row.item.parent_id
            for key, row in self.rows.items()
            if key in self.live
            and key[0] == space
            and row.ref.entry_id in eligible
            and row.item.parent_type == "memcell"
        }
        assert set(selected) == expected
        assert len(selected) == len(set(selected))
        event("selected live sources" if expected else "no live sources")

    def teardown(self) -> None:
        self.runner.close()
        self.tmp.cleanup()


TestProfileSources = ProfileSourcesMachine.TestCase
TestProfileSources.settings = settings(
    max_examples=80, stateful_step_count=30, deadline=None
)
