"""P1: memory deletion against every writer that can race it, in random order.

Real markdown and SQLite stores on a temporary root; the real /add episode
pipeline, fact extraction, user clustering, Reflection merge and finish,
profile and skill extraction, and the four deletions (one session, one
owner's session, a whole user or agent, by id). Every
store call first yields for a random moment, models and embedders take a
random time, and the plan adds the environment: the process dying between
two phases (in-memory state dropped, stores rebuilt over the same files,
interrupted deletions resent and background runs redelivered), a client
cancelling a deletion, a full disk failing one rewrite.

Checked once everything settled (the stable point; between steps the files
may be mid-change):

* I1a / J1a -- nothing a returned deletion removed is in md afterwards: no
  episode of a purged memcell, no fact of a removed episode, no deleted id;
* I4 -- every entry ever written that is gone was the target of a deletion;
* J2 -- a returned by-id deletion left none of its targets in a cluster;

before every action as well (a write-back that a later deletion of the same
scope removes again is seen only between the two):

* no episode of a memcell a returned session deletion purged, and none of
  the episodes an owner had when a returned whole-owner deletion began;

and continuously, through the lock table's acquire hook and the doubles:

* lock order -- a task holding an owner lock asks for nothing but a
  markdown file or a SQLite session; a task holding a SQLite session asks
  for no asyncio lock; the kind-level "held -> wanted" graph has no cycle;
* no model call and no embedding under an owner or clustering lock.

The index layer's interleavings (fence, eraser, cascade) are enumerated
exhaustively in ``tests/unit/test_infra/test_cascade/test_fence_permutations.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.agent_memory import OpOutcome, SkillExtractionResult
from everalgo.types import AgentSkill as AlgoAgentSkill
from everalgo.types import AtomicFact as AlgoAtomicFact
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import Profile as AlgoProfile
from hypothesis import HealthCheck, example, given, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.control import currently_in_test_context

from everos.config import Settings
from everos.core.errors import DeletionIncompleteError, SessionBusyError
from everos.core.locks import LOCKS
from everos.infra.persistence.factories import build_stores
from everos.infra.persistence.markdown import MarkdownReader
from everos.memory import IngestResult
from everos.memory.events import (
    EpisodeExtracted,
    SkillClusterUpdated,
)
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.models import Cluster, MemcellRecord, mint_cluster_id
from everos.memory.ports import MemoryRef, Scope, SessionKey, Stored
from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
from everos.memory.strategies import (
    extract_agent_skill,
    extract_atomic_facts,
    extract_user_profile,
    trigger_profile_clustering,
)
from everos.memory.strategies.extract_agent_skill import _ClusterMissingError
from tests.fakes import (
    bundled_instructions,
    make_deps,
    make_memory_service,
    user_track_llm,
)
from tests.fakes.storage import make_storage

settings.register_profile(
    "ci", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "deep",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

_DEEP = os.environ.get("HYPOTHESIS_PROFILE") == "deep"
_FACTS = "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
_PROFILE = "everos.memory.strategies.extract_user_profile.ProfileExtractor"
_SKILL = "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
_TS = 1_747_216_800_000
OWNERS = ("u1", "u2", "u3")
SEEDED = ("u1", "u2")
"""u3 has no files before the race: its first episode of a session can land
while a deletion of that session scans -- what the barrier is for."""
SESSIONS = ("s1", "s2")
AGENT = "g1"
"""The one agent: skills only, written into a cluster seeded in phase 0."""
SEEN: Counter[str] = Counter()
"""Conditions the invariants depend on, counted over the whole module."""


def _event(label: str) -> None:
    SEEN[label] += 1
    if currently_in_test_context():
        _hypothesis_event(label)


def _scope(owner: str) -> Scope:
    return Scope(app_id="a", project_id="p", owner_id=owner, owner_type="user")


class _Jitter:
    """Proxy a store; every call yields for a random moment first."""

    def __init__(self, target: Any, rng: random.Random) -> None:
        self._target = target
        self._rng = rng

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(self._rng.random() * 0.003)
            return await attr(*args, **kwargs)

        return call


class _DiskFull:
    """The k-th markdown rewrite of a deletion fails like a full disk."""

    def __init__(self, at: int | None) -> None:
        self.at = at
        self.n = 0

    def wrap(self, store: Any) -> Any:
        original = store.remove_session

        async def remove_session(scope: Scope, session_id: str) -> list[MemoryRef]:
            self.n += 1
            if self.n == self.at:
                _event("env: disk full during a rewrite")
                raise OSError(28, "No space left on device")
            return await original(scope, session_id)

        store.remove_session = remove_session
        return store


@dataclasses.dataclass
class _World:
    """What every actor shares, and what the checks read."""

    root: Path
    rng: random.Random
    storage: Any = None
    stores: Any = None
    service: Any = None
    written: dict[tuple[str, str], tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    """(owner, episode entry id) -> (session, memcell id)."""
    facts: dict[tuple[str, str], str] = dataclasses.field(default_factory=dict)
    """(owner, fact entry id) -> parent episode entry id."""
    purged: set[str] = dataclasses.field(default_factory=set)
    owner_sessions: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    by_ids: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    merged: dict[tuple[str, str], set[str]] = dataclasses.field(default_factory=dict)
    """(owner, merged entry id) -> the source entry ids it was made from."""
    returned_by_ids: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    violations: list[str] = dataclasses.field(default_factory=list)
    last_purge: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    unfinished: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    """Session -> memcells an attempt purged before it failed (500 or death)."""
    known_gap: set[str] = dataclasses.field(default_factory=set)
    pending: dict[int, tuple[Any, Any, str | None]] = dataclasses.field(
        default_factory=dict
    )
    """Deletions sent and not yet answered: what a client resends after death."""
    """Memcells of deviation 7: purged by a failed attempt, the session got a
    new /add before the resend, so the resend did not see their owners."""
    returned_purged: set[str] = dataclasses.field(default_factory=set)
    """Memcells purged by a session deletion that then returned."""
    task_purge: dict[Any, set[str]] = dataclasses.field(default_factory=dict)
    owner_deleting: set[str] = dataclasses.field(default_factory=set)
    owner_gone: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    """Owner -> episodes it had when a whole-owner deletion that returned began."""
    skill_cluster: str = ""
    memcells: int = 0
    memcell_order: dict[str, tuple[str, int]] = dataclasses.field(default_factory=dict)
    disk: _DiskFull = dataclasses.field(default_factory=lambda: _DiskFull(None))

    def build(self) -> None:
        """(Re)build the stores over the root: the process (re)starts."""
        self.storage = make_storage(self.root)
        stores = build_stores(self.storage.root, self.storage.sqlite)
        self.disk.wrap(stores.episode)
        purge = stores.session_purge
        world = self

        class _RecordingPurge:
            async def purge(self, session: SessionKey) -> list[MemcellRecord]:
                # Recorded before the call: death can land after the commit
                # and before this frame resumes. Every memcell of the session
                # numbered so far is gone once the purge commits, and a purge
                # that never committed is resent until it does.
                sid, cutoff = session.session_id, world.memcells
                ids = {
                    m
                    for m, (s, n) in world.memcell_order.items()
                    if s == sid and n <= cutoff and m not in world.purged
                }
                world.purged |= ids
                world.last_purge[sid] = ids
                world.task_purge[asyncio.current_task()] = ids
                records = await purge.purge(session)
                failed = world.unfinished.get(sid)
                if records and failed:
                    # A resend that sees new memcells takes its owners from
                    # them alone: the failed attempt's owners are missed.
                    world.known_gap |= failed
                    _event("known gap (deviation 7): resend saw new memcells")
                return records

        self.stores = dataclasses.replace(
            stores,
            episode=_Jitter(stores.episode, self.rng),
            atomic_fact=_Jitter(stores.atomic_fact, self.rng),
            memcell=_Jitter(stores.memcell, self.rng),
            cluster=_Jitter(stores.cluster, self.rng),
            session_purge=_RecordingPurge(),
        )
        self.service = make_memory_service(
            settings=Settings(), stores=self.stores, index_eraser=_NoIndex()
        )
        deleter = self.service._deleter
        delete = deleter.delete

        async def recording(**scope: Any) -> Any:
            # A session deletion that stops after its purge -- 500, death, or
            # in the background after the client went away -- leaves memcells
            # whose owners only it knew (deviation 7).
            sid = scope["session_id"]
            session_only = (
                sid is not None
                and scope["user_id"] is None
                and scope["agent_id"] is None
            )
            try:
                result = await delete(**scope)
            except (DeletionIncompleteError, asyncio.CancelledError):
                world.task_purge.pop(asyncio.current_task(), None)
                if session_only:
                    world.unfinished.setdefault(sid, set()).update(
                        world.last_purge.get(sid, set())
                    )
                raise
            purged = world.task_purge.pop(asyncio.current_task(), set())
            if session_only:
                world.returned_purged |= purged
            return result

        deleter.delete = recording


class _NoIndex:
    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        await asyncio.sleep(0)

    async def erase_owner(self, scope: Scope, kinds: Any) -> None:
        await asyncio.sleep(0)


def _watch_locks(world: _World) -> None:
    """The acquire hook: order rules, and the kind-level graph's edges."""
    edges: set[tuple[str, str]] = set()

    def kind(lock_kind: str) -> str:
        return lock_kind

    def hook(
        task: Any, wanted: tuple[str, str], held: frozenset[tuple[str, str]]
    ) -> None:
        held_kinds = {k for k, _ in held}
        if "owner" in held_kinds and wanted[0] not in ("md_path", "sqlite_session"):
            world.violations.append(f"{wanted} asked for under an owner lock {held}")
        if ("sqlite_session" in held_kinds and wanted[0] in ("session", "owner")) or (
            "sqlite_session" in held_kinds and wanted[0].startswith("partition:")
        ):
            world.violations.append(f"{wanted} asked for under a SQLite session")
        for k in held_kinds:
            if k != wanted[0] and (kind(k), kind(wanted[0])) not in edges:
                edges.add((kind(k), kind(wanted[0])))
                _event(f"lock edge: {kind(k)} -> {kind(wanted[0])}")
        if _cyclic(edges):
            world.violations.append(f"lock-order cycle: {sorted(edges)}")

    LOCKS.set_acquire_hook(hook)


def _cyclic(edges: set[tuple[str, str]]) -> bool:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for nxt in graph[node]:
            if state.get(nxt) == 1 or (nxt not in state and visit(nxt)):
                return True
        state[node] = 2
        return False

    return any(node not in state and visit(node) for node in list(graph))


def _no_lock_for_models(world: _World, what: str) -> None:
    held = {k for k, _ in LOCKS.held()}
    if "owner" in held or "partition:trigger_profile_clustering" in held:
        world.violations.append(f"{what} called holding {sorted(held)}")


class _Embedder:
    dim = 4

    def __init__(self, world: _World) -> None:
        self.world = world

    async def embed(self, text: str) -> list[float]:
        _no_lock_for_models(self.world, "embedding")
        await asyncio.sleep(self.world.rng.random() * 0.004)
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)


# ── actors ────────────────────────────────────────────────────────────────


async def _add(world: _World, session: str, owners: tuple[str, ...]) -> None:
    """/add: the memcell row under the session lock, then the episode write."""
    cell = MemCell(
        items=[
            ChatMessage(
                id=f"m{o}",
                role="user",
                content="x",
                timestamp=_TS,
                sender_id=o,
            )
            for o in owners
        ],
        timestamp=_TS,
    )
    key = SessionKey(app_id="a", project_id="p", session_id=session)
    async with LOCKS.hold("session", f"a\x1fp\x1f{session}"):
        # Numbered under the lock a session deletion's purge also holds, so
        # "inserted before the purge" is "numbered at most its cutoff".
        world.memcells += 1
        memcell_id = f"mc_{world.memcells}"
        world.memcell_order[memcell_id] = (session, world.memcells)
        await world.stores.memcell.insert_many(
            key,
            [
                MemcellRecord(
                    memcell_id=memcell_id,
                    cell=cell,
                    raw_type="conversation",
                    message_ids=[],
                    sender_ids=list(owners),
                )
            ],
        )
    llm = user_track_llm()
    pipeline = UserMemoryPipeline(
        episode_store=world.stores.episode,
        memcell_store=world.stores.memcell,
        instructions=bundled_instructions(),
        episode_llm=llm,
        summary_llm=llm,
    )
    # The extraction model; sometimes slow enough for a whole session
    # deletion to land between the memcell row and the episode write.
    await asyncio.sleep(world.rng.choice((0.001, 0.004, 0.06)))
    outcome = await pipeline.run_cell(
        IngestResult(session_id=session, messages=[], app_id="a", project_id="p"),
        PreparedCell(cell=cell, memcell_id=memcell_id, sender_ids=list(owners)),
    )
    for ev in outcome.events:
        if isinstance(ev, EpisodeExtracted):
            world.written[(ev.owner_id, ev.episode_entry_id)] = (session, memcell_id)
    if len(outcome.events) == 1:
        _event("add: episode skipped, memcell gone")


async def _facts(world: _World, owner: str, entry: str | None = None) -> None:
    """Fact extraction for one of the owner's episodes (redelivery included)."""
    mine = [(e, m) for (o, e), m in world.written.items() if o == owner]
    if not mine:
        return
    if entry is not None:
        mine = [(e, m) for e, m in mine if e == entry]
    entry, (session, memcell) = world.rng.choice(mine)
    deps = make_deps(stores=world.stores, embedding=_Embedder(world))
    event = EpisodeExtracted(
        memcell_id=memcell,
        episode_entry_id=entry,
        episode_text="t",
        episode_timestamp_ms=_TS,
        owner_id=owner,
        session_id=session,
        app_id="a",
        project_id="p",
    )
    (out,) = await extract_atomic_facts(event, deps)
    for fact_id in out.fact_entry_ids:
        world.facts[(owner, fact_id)] = entry
    if not out.fact_entry_ids:
        _event("facts: skipped, source gone")


async def _cluster(world: _World, owner: str, entry: str | None = None) -> None:
    mine = [(e, m) for (o, e), m in world.written.items() if o == owner]
    if not mine:
        return
    if entry is not None:
        mine = [(e, m) for e, m in mine if e == entry]
    entry, (session, memcell) = world.rng.choice(mine)
    deps = make_deps(stores=world.stores, embedding=_Embedder(world))
    await trigger_profile_clustering(
        EpisodeExtracted(
            memcell_id=memcell,
            episode_entry_id=entry,
            episode_text="t",
            episode_timestamp_ms=_TS,
            owner_id=owner,
            session_id=session,
            app_id="a",
            project_id="p",
        ),
        deps,
    )


class _MdIndex:
    """The episode index as the md files (what the cascade converges to)."""

    def __init__(self, world: _World) -> None:
        self.world = world

    async def orphans(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[Any]]:
        return await self.world.stores.episode.read_many(scope, entry_ids)


class _NoFacts:
    async def find(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


async def _reflect(world: _World, owner: str) -> None:
    """A Reflection merge of the owner's clusters, then its finish."""
    stores = world.stores
    rng = world.rng

    async def areflect(*args: Any, **kwargs: Any) -> AlgoEpisode:
        _no_lock_for_models(world, "reflection model")
        await asyncio.sleep(rng.random() * 0.006)
        return AlgoEpisode(
            owner_id=owner, episode="merged", summary="m", subject="s", timestamp=_TS
        )

    reflector = AsyncMock()
    reflector.areflect = areflect
    orchestrator = ReflectionOrchestrator(
        cluster_store=stores.cluster,
        episode_store=stores.episode,
        episode_index=_MdIndex(world),  # type: ignore[arg-type]
        memcell_store=stores.memcell,
        report_store=stores.reflection_report,
        reflector=reflector,
    )
    merged = await orchestrator.run(scope=_scope(owner))
    if not merged:
        return
    _event("reflect: merged")
    finisher = ReflectionFinisher(
        cluster_store=stores.cluster,
        episode_store=stores.episode,
        atomic_fact_store=stores.atomic_fact,
        atomic_fact_index=_NoFacts(),  # type: ignore[arg-type]
        report_store=stores.reflection_report,
        embedder=_Embedder(world),  # type: ignore[arg-type]
    )
    for ev in merged:
        world.written[(owner, ev.episode_entry_id)] = ("", ev.episode_entry_id)
        assert ev.reflection is not None
        world.merged[(owner, ev.episode_entry_id)] = {
            e.entry_id for e in ev.reflection.episodes
        }
        await finisher.finish(
            scope=_scope(owner),
            merged_entry_id=ev.episode_entry_id,
            context=ev.reflection,
        )


async def _delete_session(world: _World, session: str, cancel: bool) -> None:
    def finished() -> None:
        _event("delete: session returned")
        world.unfinished.pop(session, None)

    await _settle_deletion(
        world,
        lambda: world.service.delete(
            app_id="a", project_id="p", user_id=None, agent_id=None, session_id=session
        ),
        cancel,
        finished,
        session=session,
    )


async def _delete_owner_session(world: _World, owner: str, session: str) -> None:
    async def run() -> None:
        await world.service.delete(
            app_id="a", project_id="p", user_id=owner, agent_id=None, session_id=session
        )
        world.owner_sessions.add((owner, session))

    await _settle_deletion(world, run, False, lambda: _event("delete: owner session"))


async def _delete_by_id(world: _World, owner: str) -> None:
    mine = [e for (o, e) in world.written if o == owner]
    if not mine:
        return
    target = world.rng.choice(mine)
    world.by_ids.add((owner, target))

    async def run() -> None:
        await world.service.delete_by_ids(
            app_id="a",
            project_id="p",
            memory_type="episode",
            ids=[f"{owner}_{target}"],
            reason_code=None,
            note=None,
        )
        world.returned_by_ids.add((owner, target))

    await _settle_deletion(world, run, False, lambda: _event("delete: by id"))


async def _settle_deletion(
    world: _World, make: Any, cancel: bool, done: Any, session: str | None = None
) -> None:
    """Run a deletion as a client would: a 503 is an answer; a 500 is resent
    (the same request finishes it); a cancelled one runs on, shielded."""
    token = id(make)
    world.pending[token] = (make, done, session)
    for _ in range(3):
        task = asyncio.ensure_future(make())
        if cancel:
            cancel = False
            await asyncio.sleep(world.rng.random() * 0.004)
            task.cancel()
            _event("env: client cancelled a deletion")
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # the process died: phase 2 resends it
            await world.service._extractions.drain(10)
            continue  # the client sends it again
        except SessionBusyError:
            _event("delete: busy (503)")
            world.pending.pop(token, None)
            return
        except DeletionIncompleteError:
            _event("delete: incomplete (500), resent")
            continue
        world.pending.pop(token, None)
        done()
        return
    raise AssertionError("a deletion did not finish in three sends")


# ── plan and run ──────────────────────────────────────────────────────────

_ACTIONS = st.one_of(
    st.tuples(
        st.just("add"),
        st.sampled_from(SESSIONS),
        st.sampled_from([("u1",), ("u2",), ("u3",), ("u1", "u2"), ("u2", "u3")]),
    ),
    st.tuples(st.just("facts"), st.sampled_from(OWNERS)),
    st.tuples(st.just("cluster"), st.sampled_from(OWNERS)),
    st.tuples(st.just("reflect"), st.sampled_from(OWNERS)),
    st.tuples(st.just("del_session"), st.sampled_from(SESSIONS), st.booleans()),
    st.tuples(
        st.just("del_owner_session"), st.sampled_from(OWNERS), st.sampled_from(SESSIONS)
    ),
    st.tuples(st.just("del_id"), st.sampled_from(OWNERS)),
    st.tuples(st.just("del_owner"), st.sampled_from((*OWNERS, AGENT))),
    st.tuples(st.just("profile"), st.sampled_from(OWNERS)),
    st.tuples(st.just("skill")),
)


async def _profile(world: _World, owner: str) -> None:
    """Profile extraction from one of the owner's episodes: its partition lock,
    the model under it, then the owner lock for the write."""
    mine = [(e, m) for (o, e), m in world.written.items() if o == owner]
    if not mine:
        return
    entry, (session, memcell) = world.rng.choice(mine)
    deps = make_deps(stores=world.stores, embedding=_Embedder(world))
    await extract_user_profile(
        EpisodeExtracted(
            memcell_id=memcell,
            episode_entry_id=entry,
            episode_text="t",
            episode_timestamp_ms=_TS,
            owner_id=owner,
            session_id=session,
            app_id="a",
            project_id="p",
        ),
        deps,
    )
    _event("profile: ran")


async def _skill(world: _World) -> None:
    """Skill extraction into the agent's seeded cluster (gone once the agent
    was deleted: OME would retry, then dead-letter)."""
    deps = make_deps(stores=world.stores, embedding=_Embedder(world))
    try:
        await extract_agent_skill(
            SkillClusterUpdated(
                case_entry_id="ac_1",
                cluster_id=world.skill_cluster,
                agent_id=AGENT,
                app_id="a",
                project_id="p",
            ),
            deps,
        )
    except _ClusterMissingError:
        _event("skill: cluster gone")
        return
    _event("skill: ran")


async def _delete_owner(world: _World, owner: str) -> None:
    user, agent = (None, owner) if owner == AGENT else (owner, None)

    async def run() -> None:
        world.owner_deleting.add(owner)
        before = {e for (o, e) in world.written if o == owner}
        await world.service.delete(
            app_id="a", project_id="p", user_id=user, agent_id=agent, session_id=None
        )
        world.owner_gone.setdefault(owner, set()).update(before)

    await _settle_deletion(world, run, False, lambda: _event("delete: owner"))


def _start(world: _World, action: tuple[Any, ...]) -> Any:
    kind, *args = action
    return {
        "add": _add,
        "facts": _facts,
        "cluster": _cluster,
        "reflect": _reflect,
        "del_session": _delete_session,
        "del_owner_session": _delete_owner_session,
        "del_id": _delete_by_id,
        "del_owner": _delete_owner,
        "profile": _profile,
        "skill": _skill,
        "resend": lambda w, make, done, session: _settle_deletion(
            w, make, False, done, session
        ),
    }[kind](world, *args)


async def _phase(
    world: _World, actions: Sequence[tuple[Any, ...]], crash_after: float | None
) -> None:
    async def delayed(action: tuple[Any, ...]) -> None:
        await asyncio.sleep(world.rng.random() * 0.01)
        await _no_write_back(world)
        await _start(world, action)

    tasks = [asyncio.ensure_future(delayed(a)) for a in actions]
    if crash_after is None:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        await asyncio.sleep(crash_after)
        # Death: every task stops where it is, shielded deletions included.
        running = [*tasks, *world.service._extractions._tasks]
        for t in running:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*running, return_exceptions=True)
        _event("env: process died mid-phase")
    for r in results:
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r


async def _md_entries(
    world: _World,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], str]]:
    """Every episode (owner, id) -> inline, and every fact (owner, id) -> parent."""
    episodes: dict[tuple[str, str], dict[str, str]] = {}
    facts: dict[tuple[str, str], str] = {}
    for md in world.root.rglob("*.md"):
        if ".tmp" in md.parts or md.parent.name not in ("episodes", ".atomic_facts"):
            continue
        owner = md.parent.parent.name
        try:
            doc = await MarkdownReader.read(md)
        except FileNotFoundError:  # a whole-owner deletion moved it meanwhile
            continue
        for e in doc.entries:
            inline = dict(e.as_structured().inline)
            if md.parent.name == "episodes":
                episodes[(owner, e.id)] = inline
            elif md.parent.name == ".atomic_facts":
                facts[(owner, e.id)] = inline.get("parent_id", "")
    return episodes, facts


async def _seed(world: _World) -> None:
    """Phase 0, in order: episodes in both sessions, clustered, with facts --
    something for every later actor to race over."""
    for i in range(4):
        for owner in SEEDED:
            await _add(world, SESSIONS[i % 2], (owner,))
    for owner, entry in list(world.written):
        await _cluster(world, owner, entry)
        await _facts(world, owner, entry)
    world.skill_cluster = mint_cluster_id()
    await world.stores.cluster.upsert(
        Scope(app_id="a", project_id="p", owner_id=AGENT, owner_type="agent"),
        "agent_case",
        Cluster(id=world.skill_cluster, centroid=(1.0, 0.0, 0.0, 0.0), last_ts=_TS),
        member_type="case",
    )


def _closure(world: _World, owner: str, target: str) -> set[str]:
    """A by-id target and everything merged into it, transitively."""
    out, todo = {target}, [target]
    while todo:
        for src in world.merged.get((owner, todo.pop()), set()) - out:
            out.add(src)
            todo.append(src)
    return out


async def _no_write_back(world: _World) -> None:
    """What a returned session or whole-owner deletion removed is not in md,
    at any moment (md files are replaced whole, so any moment is readable)."""
    episodes, _ = await _md_entries(world)
    for (owner, entry), inline in episodes.items():
        assert inline.get("parent_id") not in world.returned_purged, (
            "written back after a session deletion",
            owner,
            entry,
        )
        assert entry not in world.owner_gone.get(owner, set()), (
            "written back after an owner deletion",
            owner,
            entry,
        )


async def _check(world: _World) -> None:
    assert not world.violations, world.violations
    await _no_write_back(world)
    episodes, facts = await _md_entries(world)
    # I1a: no episode of a purged memcell; no fact of a missing episode.
    for (owner, entry), inline in episodes.items():
        if inline.get("parent_id") in world.known_gap:
            continue  # deviation 7, open: reported, not fixed
        assert inline.get("parent_id") not in world.purged, (owner, entry, inline)
    for (owner, fact), parent in facts.items():
        if world.written.get((owner, parent), ("", ""))[1] in world.known_gap:
            continue  # deviation 7
        assert (owner, parent) in episodes, ("fact outlived its episode", owner, fact)
    # J1a / J2: a returned by-id deletion left its closure nowhere.
    clusters = world.stores.cluster
    for owner, target in world.returned_by_ids:
        for entry in _closure(world, owner, target):
            assert (owner, entry) not in episodes, ("by-id closure in md", owner, entry)
            found = await clusters.find_cluster_for_member(
                _scope(owner), "episode", entry
            )
            assert found is None, ("by-id closure still clustered", owner, entry)
    # I4: whatever was written and is gone was some deletion's target.
    by_id = {(o, e) for o, t in world.by_ids for e in _closure(world, o, t)}
    for (owner, entry), (session, memcell) in world.written.items():
        if (owner, entry) in episodes:
            continue
        assert (
            memcell in world.purged
            or (owner, session) in world.owner_sessions
            or (owner, entry) in by_id
            or owner in world.owner_deleting
        ), ("an episode vanished with no deletion", owner, entry)
    for (owner, fact), parent in world.facts.items():
        if (owner, fact) not in facts and owner not in world.owner_deleting:
            assert (owner, parent) not in episodes or world.written[(owner, parent)][
                1
            ] in world.purged, (
                "a fact vanished with its episode still there",
                owner,
                fact,
            )
    if world.purged:
        _event("check: a session deletion purged memcells")


@given(
    first=st.lists(_ACTIONS, min_size=4, max_size=12),
    second=st.lists(_ACTIONS, min_size=2, max_size=8),
    seed=st.integers(0, 2**16),
    crash=st.one_of(st.none(), st.floats(0.001, 0.02)),
    disk_full_at=st.one_of(st.none(), st.integers(1, 3)),
)
@example(  # an interleaving a review run reported as failing; kept on every run
    first=[("cluster", "u2"), ("del_id", "u2"), ("reflect", "u2"), ("skill",)],
    second=[("add", "s1", ("u1",)), ("facts", "u1")],
    seed=0,
    crash=None,
    disk_full_at=None,
)
def test_deletions_and_writers_keep_their_invariants(
    first: list[tuple[Any, ...]],
    second: list[tuple[Any, ...]],
    seed: int,
    crash: float | None,
    disk_full_at: int | None,
) -> None:
    # A10's bound: every run ends within 30 s, or something is stuck.
    asyncio.run(asyncio.wait_for(_run(first, second, seed, crash, disk_full_at), 30))


async def _run(
    first: list[tuple[Any, ...]],
    second: list[tuple[Any, ...]],
    seed: int,
    crash: float | None,
    disk_full_at: int | None,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="delete_jitter_"))
    LOCKS.reset()
    world = _World(root=tmp, rng=random.Random(seed), disk=_DiskFull(disk_full_at))
    try:
        world.build()
        await world.storage.create_schema()
        _watch_locks(world)
        facts = [AlgoAtomicFact(owner_id=None, content="f", timestamp=_TS)]
        with (
            patch(_FACTS) as extractor_cls,
            patch(_PROFILE) as profile_cls,
            patch(_SKILL) as skill_cls,
        ):

            async def extract(*args: Any, **kwargs: Any) -> list[AlgoAtomicFact]:
                _no_lock_for_models(world, "fact model")
                await asyncio.sleep(world.rng.random() * 0.004)
                return facts

            async def profile(*args: Any, **kwargs: Any) -> AlgoProfile:
                await asyncio.sleep(world.rng.random() * 0.004)
                return AlgoProfile(
                    owner_id=kwargs["owner_id"], summary="p", timestamp=_TS
                )

            async def skills(*args: Any, **kwargs: Any) -> SkillExtractionResult:
                await asyncio.sleep(world.rng.random() * 0.004)
                skill = AlgoAgentSkill(id="k", name="tidy", content="c", confidence=0.5)
                return SkillExtractionResult(None, {}, [OpOutcome(0, skill, None, {})])

            extractor_cls.return_value.aextract_from_text = extract
            profile_cls.return_value.aextract_from_episodes = profile
            skill_cls.return_value.aextract_with_reason = skills
            await _seed(world)
            await _phase(world, first, crash)
            if crash is not None:  # restart: nothing in memory survives
                await world.storage.aclose()
                LOCKS.reset()
                world.build()
                _watch_locks(world)
                # Interrupted deletions are resent as they were sent.
                resends = list(world.pending.values())
                world.pending.clear()
                second = [("resend", *r) for r in resends] + second
            await _phase(world, second, None)
        await _check(world)
    finally:
        LOCKS.reset()
        if world.storage is not None:
            await world.storage.aclose()
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def _conditions_were_reached() -> Iterator[None]:
    """A condition an invariant guards that never happened is a false pin."""
    yield
    if _DEEP:
        needed = [
            "delete: session returned",
            "delete: by id",
            "delete: owner",
            "profile: ran",
            "skill: ran",
            "add: episode skipped, memcell gone",
            "facts: skipped, source gone",
            "reflect: merged",
            "env: process died mid-phase",
            "env: client cancelled a deletion",
            "env: disk full during a rewrite",
            "check: a session deletion purged memcells",
        ]
        missing = [c for c in needed if not SEEN[c]]
        assert not missing, (missing, dict(SEEN))
