"""Tests for :func:`extract_user_profile`.

The strategy reads the event's episode back from md and its memcell from the
memcell store, adds the owner's waiting episodes the index holds, and runs
``ProfileExtractor.aextract_from_episodes`` (patched) once on them all.
In-memory doubles from :mod:`tests.fakes` stand in for every port, so the
assertions are on what was passed, stored and marked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile
from everalgo.utils import InvalidInputError as AlgoInvalidInputError

from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.models import Episode, EpisodeParentType, Profile
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.profile_batcher import ProfileBatcher
from everos.memory.strategies import EXTRACT_USER_PROFILE_SPEC, extract_user_profile
from tests.fakes import make_deps

_EXTRACTOR = "everos.memory.strategies.extract_user_profile.ProfileExtractor"
_TS = 1_700_000_001_000


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    episode_entry_id: str = "ep_20260101_0001",
    owner_name: str | None = "Alice",
    source: str = "pipeline",
    language: str | None = None,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=episode_entry_id,
        episode_text="as carried on the event",
        episode_timestamp_ms=_TS,
        owner_id=owner_id,
        owner_name=owner_name,
        source=source,
        language=language,
    )


def _seed(
    deps: MemoryDeps,
    scope: Scope,
    *,
    entry_id: str = "ep_20260101_0001",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    narrative: str = "Alice plans a trip to Kyoto.",
    timestamp: int = _TS,
) -> None:
    """The owner's copy of an episode in md, and its memcell."""
    deps.stores.episode.seed(
        scope,
        entry_id,
        Episode(
            owner_id=scope.owner_id,
            episode=narrative,
            summary="Kyoto trip",
            subject="Trip",
            timestamp=timestamp,
            parent_id=memcell_id,
        ),
    )
    line = ChatMessage(
        id=f"{memcell_id}_m0",
        role="user",
        content="hi",
        timestamp=timestamp,
        sender_id=scope.owner_id,
    )
    deps.stores.memcell.seed(memcell_id, MemCell(items=[line], timestamp=timestamp))


@contextmanager
def _extractor(**mock: object) -> Iterator[AsyncMock]:
    """Patch ``aextract_from_episodes`` with an ``AsyncMock`` configured by ``mock``."""
    extract = AsyncMock(**mock)
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_episodes = extract
        yield extract


def _algo_profile(owner_id: str = "u_alice", **fields: object) -> AlgoProfile:
    return AlgoProfile.model_validate(
        {"owner_id": owner_id, "summary": "next", "timestamp": _TS, **fields}
    )


def test_spec() -> None:
    spec = EXTRACT_USER_PROFILE_SPEC
    assert spec.name == "extract_user_profile"
    assert spec.trigger.on == [EpisodeExtracted]
    assert spec.emits == frozenset()
    assert spec.max_retries == 2


@pytest.mark.parametrize(
    ("source", "applies"), [("pipeline", True), ("reflection", False)]
)
def test_only_pipeline_episodes_are_profiled(source: str, applies: bool) -> None:
    assert (
        EXTRACT_USER_PROFILE_SPEC.applies_to(_event(source=source), make_deps())
        is applies
    )


async def test_init_extracts_from_the_episode_read_back_from_md() -> None:
    """No prior profile: the stored narrative is passed, not the event's copy."""
    deps = make_deps()
    _seed(deps, _scope())
    new_profile = _algo_profile(
        summary="Alice is a traveller.",
        explicit_info=[{"description": "plans a trip to Kyoto"}],
        implicit_traits=[{"description": "adventurous"}],
    )
    with _extractor(return_value=new_profile) as extract:
        events = await extract_user_profile(_event(), deps)

    assert events == []
    (episodes,), kwargs = extract.call_args
    assert [(e.episode, e.timestamp) for e in episodes] == [
        ("Alice plans a trip to Kyoto.", _TS)
    ]
    assert kwargs["owner_id"] == "u_alice"
    assert kwargs["owner_name"] == "Alice"
    assert kwargs["old_profile"] is None
    stored = deps.stores.profile.profiles[_scope()]
    assert stored.summary == "Alice is a traveller."
    assert [i.description for i in stored.explicit_info] == ["plans a trip to Kyoto"]
    assert [i.description for i in stored.implicit_traits] == ["adventurous"]


async def test_the_extractor_runs_on_the_profile_llm() -> None:
    profile_llm = object()
    deps = make_deps(profile_llm=profile_llm)
    _seed(deps, _scope())
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_episodes = AsyncMock(
            return_value=_algo_profile()
        )
        await extract_user_profile(_event(), deps)

    assert deps.llm is not profile_llm
    extractor_cls.assert_called_once_with(llm=profile_llm)


async def test_update_rehydrates_the_stored_profile() -> None:
    deps = make_deps()
    _seed(deps, _scope())
    deps.stores.profile.profiles[_scope()] = Profile(
        owner_id="u_alice",
        summary="prior summary",
        timestamp=_TS - 1,
        explicit_info=["prior fact"],
    )
    with _extractor(return_value=_algo_profile(summary="updated summary")) as extract:
        await extract_user_profile(_event(), deps)

    old = extract.call_args.kwargs["old_profile"]
    assert isinstance(old, AlgoProfile)
    assert (old.summary, old.timestamp) == ("prior summary", _TS - 1)
    assert [i.description for i in old.explicit_info] == ["prior fact"]
    assert deps.stores.profile.profiles[_scope()].summary == "updated summary"


# ── waiting episodes: folded along, then skipped by their own runs ──────


def _seed_indexed(
    deps: MemoryDeps,
    scope: Scope,
    entry_id: str,
    *,
    timestamp: int,
    in_md: bool = True,
    parent_type: EpisodeParentType = "memcell",
) -> None:
    """An owner's episode the index holds; its md copy and memcell unless not."""
    memcell_id = f"mc_{entry_id}"
    narrative = f"Alice wrote {entry_id}."
    if in_md:
        _seed(
            deps,
            scope,
            entry_id=entry_id,
            memcell_id=memcell_id,
            narrative=narrative,
            timestamp=timestamp,
        )
    ref = MemoryRef(
        id=f"{scope.owner_id}_{entry_id}",
        entry_id=entry_id,
        md_path=f"users/{scope.owner_id}/episodes/2026-01-01.md",
    )
    episode = Episode(
        owner_id=scope.owner_id,
        episode=narrative,
        summary="",
        timestamp=timestamp,
        parent_id=memcell_id,
        parent_type=parent_type,
    )
    deps.indexes.episode.seed(Stored(ref=ref, scope=scope, item=episode))


def _own_event(entry_id: str) -> EpisodeExtracted:
    return _event(episode_entry_id=entry_id, memcell_id=f"mc_{entry_id}")


def _passed(extract: AsyncMock) -> list[str]:
    """The narratives of the last call, oldest first."""
    (episodes,), _ = extract.call_args
    return [e.episode for e in sorted(episodes, key=lambda e: e.timestamp)]


async def test_waiting_episodes_ride_along_and_their_runs_make_no_call() -> None:
    deps = make_deps()
    _seed_indexed(deps, _scope(), "ep_own", timestamp=_TS)
    _seed_indexed(deps, _scope(), "ep_w1", timestamp=_TS + 1)
    _seed_indexed(deps, _scope(), "ep_w2", timestamp=_TS + 2)
    with _extractor(return_value=_algo_profile(timestamp=_TS + 2)) as extract:
        await extract_user_profile(_own_event("ep_own"), deps)
        await extract_user_profile(_own_event("ep_w1"), deps)
        await extract_user_profile(_own_event("ep_w2"), deps)

    assert extract.call_count == 1
    assert _passed(extract) == [
        "Alice wrote ep_own.",
        "Alice wrote ep_w1.",
        "Alice wrote ep_w2.",
    ]
    assert not deps.profile_batcher.is_marked(_scope(), "ep_w1")


async def test_the_oldest_waiting_episodes_fill_the_cap() -> None:
    deps = make_deps(profile_batcher=ProfileBatcher(3))
    _seed(deps, _scope(), entry_id="ep_own", memcell_id="mc_ep_own")
    for entry_id, offset in [("ep_d", 4), ("ep_a", 1), ("ep_c", 3), ("ep_b", 2)]:
        _seed_indexed(deps, _scope(), entry_id, timestamp=_TS + offset)
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_own_event("ep_own"), deps)

    assert _passed(extract) == [
        "Alice plans a trip to Kyoto.",
        "Alice wrote ep_a.",
        "Alice wrote ep_b.",
    ]


async def test_a_cap_of_one_folds_the_events_episode_alone() -> None:
    deps = make_deps(profile_batcher=ProfileBatcher(1))
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_w1", timestamp=_TS + 1)
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(), deps)

    assert _passed(extract) == ["Alice plans a trip to Kyoto."]
    assert not deps.profile_batcher.is_marked(_scope(), "ep_w1")


async def test_an_episode_older_than_the_profile_is_left_to_its_own_run() -> None:
    deps = make_deps()
    _seed(deps, _scope(), timestamp=_TS + 10)
    deps.stores.profile.profiles[_scope()] = Profile(
        owner_id="u_alice", summary="prior", timestamp=_TS + 5
    )
    _seed_indexed(deps, _scope(), "ep_late", timestamp=_TS + 1)
    _seed_indexed(deps, _scope(), "ep_new", timestamp=_TS + 20)
    with _extractor(return_value=_algo_profile(timestamp=_TS + 20)) as extract:
        await extract_user_profile(_event(), deps)
        assert _passed(extract) == [
            "Alice plans a trip to Kyoto.",
            "Alice wrote ep_new.",
        ]
        await extract_user_profile(_own_event("ep_late"), deps)

    assert extract.call_count == 2
    assert _passed(extract) == ["Alice wrote ep_late."]


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param({"in_md": False}, id="gone-from-md"),
        pytest.param({"parent_type": "cluster"}, id="reflection-merged"),
    ],
)
async def test_an_unfoldable_index_row_is_left_out(missing: dict[str, object]) -> None:
    deps = make_deps()
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_x", timestamp=_TS + 1, **missing)  # type: ignore[arg-type]
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(), deps)

    assert _passed(extract) == ["Alice plans a trip to Kyoto."]
    assert not deps.profile_batcher.is_marked(_scope(), "ep_x")


async def test_a_waiting_episode_whose_memcell_is_gone_is_left_out() -> None:
    deps = make_deps()
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_x", timestamp=_TS + 1)
    del deps.stores.memcell.cells["mc_ep_x"]
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(), deps)

    assert _passed(extract) == ["Alice plans a trip to Kyoto."]


async def test_an_episode_folded_ahead_is_not_folded_again() -> None:
    """Its own run has not come yet; another run must not send it twice."""
    deps = make_deps()
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_w1", timestamp=_TS + 1)
    deps.profile_batcher.mark(_scope(), ["ep_w1"])
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(), deps)

    assert _passed(extract) == ["Alice plans a trip to Kyoto."]


async def test_a_failed_call_marks_nothing() -> None:
    deps = make_deps()
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_w1", timestamp=_TS + 1)
    with (
        _extractor(side_effect=RuntimeError("gateway down")),
        pytest.raises(RuntimeError),
    ):
        await extract_user_profile(_event(), deps)

    assert not deps.profile_batcher.is_marked(_scope(), "ep_w1")


async def test_a_refused_batch_still_marks_its_waiting_episodes() -> None:
    """A retry would read the same narratives, so their runs need not call."""
    deps = make_deps()
    _seed(deps, _scope())
    _seed_indexed(deps, _scope(), "ep_w1", timestamp=_TS + 1)
    refusal = AlgoInvalidInputError("no episode narrative mentions 'Alice'")
    with _extractor(side_effect=refusal):
        await extract_user_profile(_event(), deps)

    assert deps.stores.profile.writes == []
    assert deps.profile_batcher.is_marked(_scope(), "ep_w1")


# ── owner_name: the label the event carries ──────────────────────────────


@pytest.mark.parametrize(
    ("carried", "expected"),
    [
        pytest.param("Alice", "Alice", id="event-label"),
        pytest.param(None, "u_alice", id="no-label-falls-back-to-id"),
    ],
)
async def test_owner_name_is_the_label_on_the_event(
    carried: str | None, expected: str
) -> None:
    deps = make_deps()
    _seed(deps, _scope())
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(owner_name=carried), deps)

    assert extract.call_args.kwargs["owner_name"] == expected


async def test_each_owner_copy_of_one_narrative_updates_only_its_profile() -> None:
    """One memcell with two user senders: two episode copies, two profiles."""
    deps = make_deps()
    narrative = "Alice and Bob plan a trip to Kyoto."
    for owner, entry in (("u_alice", "ep_a"), ("u_bob", "ep_b")):
        _seed(deps, _scope(owner), entry_id=entry, narrative=narrative)

    async def extract(_episodes, *, owner_id, owner_name, **_kwargs):
        return _algo_profile(owner_id, summary=f"profile of {owner_name}")

    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_episodes = extract
        await extract_user_profile(_event(episode_entry_id="ep_a"), deps)
        await extract_user_profile(
            _event(owner_id="u_bob", episode_entry_id="ep_b", owner_name="Bob"), deps
        )

    profiles = deps.stores.profile.profiles
    assert profiles[_scope("u_alice")].summary == "profile of Alice"
    assert profiles[_scope("u_bob")].summary == "profile of Bob"


# ── skips ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("gone", ["episode", "memcell"])
async def test_a_deleted_source_is_not_profiled(gone: str) -> None:
    deps = make_deps()
    _seed(deps, _scope())
    if gone == "episode":
        deps.stores.episode.records.clear()
    else:
        deps.stores.memcell.cells.clear()
    with _extractor() as extract:
        assert await extract_user_profile(_event(), deps) == []

    extract.assert_not_called()
    assert deps.stores.profile.writes == []


async def test_a_narrative_the_extractor_refuses_is_skipped() -> None:
    """``owner_name`` absent from the narrative: logged, nothing written, no retry."""
    deps = make_deps()
    _seed(deps, _scope(), narrative="The user asked about the weather.")
    refusal = AlgoInvalidInputError("no episode narrative mentions 'Alice'")
    with (
        _extractor(side_effect=refusal),
        structlog.testing.capture_logs() as logs,
    ):
        assert await extract_user_profile(_event(), deps) == []

    assert deps.stores.profile.writes == []
    assert [e["event"] for e in logs if e["log_level"] == "warning"] == [
        "profile_episode_skipped"
    ]


# ── primary language (frozen once known) ─────────────────────────────────


@pytest.mark.parametrize(
    ("stored", "event_language", "expected"),
    [
        pytest.param(None, "Chinese", "Chinese", id="no-profile-init-freezes"),
        pytest.param("English", "Chinese", "English", id="frozen-wins-unchanged"),
        pytest.param("", "Chinese", "Chinese", id="unfrozen-takes-event"),
        pytest.param("", None, None, id="unknown-stays-unknown"),
    ],
)
async def test_profile_language_is_frozen_once_known(
    stored: str | None, event_language: str | None, expected: str | None
) -> None:
    """``stored``: ``None`` = no profile yet, ``""`` = one with no language."""
    deps = make_deps()
    _seed(deps, _scope())
    if stored is not None:
        deps.stores.profile.profiles[_scope()] = Profile(
            owner_id="u_alice",
            summary="prior",
            timestamp=_TS - 1,
            primary_language=stored or None,
        )
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(language=event_language), deps)

    assert extract.call_args.kwargs["output_language"] == expected
    assert deps.stores.profile.profiles[_scope()].primary_language == expected


async def test_primary_language_does_not_reach_the_algo_profile() -> None:
    """The frozen language is passed as ``output_language``, not as profile data."""
    deps = make_deps()
    _seed(deps, _scope())
    deps.stores.profile.profiles[_scope()] = Profile(
        owner_id="u_alice",
        summary="prior",
        timestamp=_TS - 1,
        primary_language="English",
    )
    with _extractor(return_value=_algo_profile()) as extract:
        await extract_user_profile(_event(), deps)

    assert (
        "primary_language" not in extract.call_args.kwargs["old_profile"].model_dump()
    )


# ── partition lock (owner_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(owner_a: str, owner_b: str) -> list[str]:
    """Drive two extract_user_profile runs and record entry/exit order."""
    log: list[str] = []

    async def slow_extract(_episodes, *, owner_id, **_kwargs):
        log.append(f"enter:{owner_id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{owner_id}")
        return _algo_profile(owner_id)

    deps = make_deps()
    events = []
    for owner, tag in ((owner_a, "a"), (owner_b, "b")):
        _seed(deps, _scope(owner), entry_id=f"ep_{tag}", memcell_id=f"mc_{tag}")
        events.append(
            _event(owner_id=owner, episode_entry_id=f"ep_{tag}", memcell_id=f"mc_{tag}")
        )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_episodes = slow_extract
        await asyncio.gather(*(extract_user_profile(e, deps) for e in events))
    return log


async def test_partition_lock_serialises_runs_on_same_owner() -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice")
    assert log == ["enter:u_alice", "leave:u_alice", "enter:u_alice", "leave:u_alice"]


async def test_partition_lock_lets_different_owners_run_in_parallel() -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob")
    assert log.index("enter:u_alice") < log.index("leave:u_bob")
    assert log.index("enter:u_bob") < log.index("leave:u_alice")


@pytest.mark.parametrize("field", ["app_id", "project_id"])
async def test_partition_lock_keeps_same_owner_in_different_spaces_independent(
    field: str,
) -> None:
    entered = 0
    both_entered = asyncio.Event()

    async def extract(_episodes, *, owner_id, **_kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        return _algo_profile(owner_id)

    deps = make_deps()
    scopes = [_scope(), replace(_scope(), **{field: "other"})]
    for scope in scopes:
        _seed(deps, scope)
    events = [_event().model_copy(update={field: getattr(s, field)}) for s in scopes]
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_episodes = extract
        async with asyncio.timeout(2):
            await asyncio.gather(*(extract_user_profile(e, deps) for e in events))
    assert len(deps.stores.profile.writes) == 2
    assert all([await deps.stores.profile.read(scope) is not None for scope in scopes])
