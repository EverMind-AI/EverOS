"""Markdown adapters of the memory truth ports, against a real memory root."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import structlog.testing

from everos.core.locks import LOCKS, owner_key
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownAgentCaseStore,
    MarkdownAgentSkillStore,
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
    MarkdownForesightStore,
    MarkdownProfileStore,
    MarkdownReader,
    parse_structured_entry,
)
from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    Profile,
)
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    AgentCaseStore,
    AgentSkillStore,
    AtomicFactStore,
    EpisodeStore,
    ForesightStore,
    ProfileStore,
    Scope,
)

_TS = 1_747_216_800_000  # 2025-05-14T10:00:00Z


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _user() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


def _agent() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="a1", owner_type="agent")


async def _entries(root: MemoryRoot, md_path: str) -> dict[str, dict]:
    parsed = await MarkdownReader.read(root.root / md_path)
    result: dict[str, dict] = {}
    for entry in parsed.entries:
        structured = parse_structured_entry(entry.body)
        result[entry.id] = {
            "inline": dict(structured.inline),
            "sections": dict(structured.sections),
        }
    return result


def _episode(**overrides: object) -> Episode:
    fields: dict[str, object] = {
        "owner_id": "u1",
        "episode": "we talked",
        "timestamp": _TS,
        "subject": "Chat",
        "summary": "short",
        "session_id": "s1",
        "sender_ids": ["u1"],
        "parent_id": "mc_1",
    }
    return Episode(**(fields | overrides))  # type: ignore[arg-type]


async def test_append_skipped_when_source_gone_returns_none_and_writes_nothing(
    root: MemoryRoot,
) -> None:
    async def gone() -> bool:
        return False

    store = MarkdownEpisodeStore(root)
    with structlog.testing.capture_logs() as logs:
        ref = await store.append(_user(), _episode(), only_if=gone)
        marked = await store.deprecate(
            _user(), entries=[("x.md", "ep_1")], replaced_by="m", only_if=gone
        )

    assert ref is None
    assert marked is False
    assert not list((root.root / "app").rglob("*.md"))
    skipped = [e for e in logs if e["event"] == "memory_write_skipped_source_gone"]
    assert [(e["kind"], e["owner_id"], e["count"]) for e in skipped] == [
        ("episode", "u1", 1),
        ("episode", "u1", 1),
    ]
    assert "we talked" not in str(skipped)


async def test_only_if_runs_inside_owner_lock(root: MemoryRoot) -> None:
    seen: list[frozenset[tuple[str, str]]] = []

    async def alive() -> bool:
        seen.append(LOCKS.held())
        return True

    ref = await MarkdownEpisodeStore(root).append(_user(), _episode(), only_if=alive)

    assert ref is not None
    assert seen == [frozenset({("owner", owner_key("app", "proj", "user", "u1"))})]


async def test_profile_and_skill_writes_hold_the_owner_lock(root: MemoryRoot) -> None:
    """A whole-owner deletion takes the same lock; these writes must too."""
    profiles = MarkdownProfileStore(root)
    skills = MarkdownAgentSkillStore(root)
    async with LOCKS.hold("owner", owner_key("app", "proj", "user", "u1")):
        blocked = asyncio.create_task(
            profiles.write(_user(), Profile(owner_id="u1", summary="s", timestamp=_TS))
        )
        await asyncio.sleep(0.05)
        assert not blocked.done()
    await blocked
    async with LOCKS.hold("owner", owner_key("app", "proj", "agent", "a1")):
        blocked = asyncio.create_task(
            skills.write(
                _agent(), AgentSkill(id="sk1", name="k", description="d", content="c")
            )
        )
        await asyncio.sleep(0.05)
        assert not blocked.done()
    await blocked


def test_adapters_satisfy_their_ports(root: MemoryRoot) -> None:
    assert isinstance(MarkdownEpisodeStore(root), EpisodeStore)
    assert isinstance(MarkdownAtomicFactStore(root), AtomicFactStore)
    assert isinstance(MarkdownForesightStore(root), ForesightStore)
    assert isinstance(MarkdownAgentCaseStore(root), AgentCaseStore)
    assert isinstance(MarkdownProfileStore(root), ProfileStore)
    assert isinstance(MarkdownAgentSkillStore(root), AgentSkillStore)


async def test_episode_append_returns_a_root_relative_ref(root: MemoryRoot) -> None:
    episode = Episode(
        owner_id="u1",
        episode="we talked",
        timestamp=_TS,
        subject="Chat",
        summary="short",
        session_id="s1",
        sender_ids=["u1", "u2"],
        parent_id="mc_1",
    )

    ref = await MarkdownEpisodeStore(root).append(
        _user(), episode, only_if=NO_SOURCE_CHECK
    )

    assert ref.id == f"u1_{ref.entry_id}"
    assert ref.md_path.startswith("app/proj/users/u1/episodes/")
    assert not Path(ref.md_path).is_absolute()
    entry = (await _entries(root, ref.md_path))[ref.entry_id]
    assert entry["inline"]["parent_type"] == "memcell"
    assert entry["inline"]["parent_id"] == "mc_1"
    assert entry["inline"]["session_id"] == "s1"
    assert entry["sections"] == {
        "Subject": "Chat",
        "Summary": "short",
        "Content": "we talked",
    }


async def test_merged_episode_has_cluster_parent_and_no_session(
    root: MemoryRoot,
) -> None:
    merged = Episode(
        owner_id="u1",
        episode="merged story",
        summary="",
        timestamp=_TS,
        subject="Topic",
        parent_id="cl_1",
        parent_type="cluster",
    )

    (ref,) = await MarkdownEpisodeStore(root).append_many(
        _user(), [merged], only_if=NO_SOURCE_CHECK
    )

    entry = (await _entries(root, ref.md_path))[ref.entry_id]
    assert entry["inline"]["parent_type"] == "cluster"
    assert "session_id" not in entry["inline"]
    assert entry["sections"] == {"Subject": "Topic", "Content": "merged story"}


async def test_episode_language_is_an_inline_field_only_when_known(
    root: MemoryRoot,
) -> None:
    """Known: written inline. Unknown: left out, as in entries written before."""
    store = MarkdownEpisodeStore(root)
    known, unknown = await store.append_many(
        _user(),
        [
            Episode(
                owner_id="u1",
                episode="我们聊了茶",
                summary="",
                timestamp=_TS,
                parent_id="mc_1",
                language="Chinese",
            ),
            Episode(
                owner_id="u1",
                episode="...",
                summary="",
                timestamp=_TS,
                parent_id="mc_2",
            ),
        ],
        only_if=NO_SOURCE_CHECK,
    )

    entries = await _entries(root, known.md_path)
    assert entries[known.entry_id]["inline"]["language"] == "Chinese"
    assert "language" not in entries[unknown.entry_id]["inline"]


async def test_episode_deprecate_patches_the_file_frontmatter(
    root: MemoryRoot,
) -> None:
    store = MarkdownEpisodeStore(root)
    ref = await store.append(
        _user(),
        Episode(
            owner_id="u1", summary="", episode="x", timestamp=_TS, parent_id="mc_1"
        ),
        only_if=NO_SOURCE_CHECK,
    )

    await store.deprecate(
        _user(),
        entries=[(ref.md_path, ref.entry_id)],
        replaced_by="ep_merged",
        only_if=NO_SOURCE_CHECK,
    )

    parsed = await MarkdownReader.read(root.root / ref.md_path)
    assert parsed.frontmatter["deprecated_entries"] == {ref.entry_id: "ep_merged"}


async def test_atomic_facts_append_and_deprecate(root: MemoryRoot) -> None:
    store = MarkdownAtomicFactStore(root)
    facts = [
        AtomicFact(owner_id="u1", content=f"fact {i}", timestamp=_TS, parent_id="ep_1")
        for i in range(2)
    ]

    refs = await store.append_many(_user(), facts, only_if=NO_SOURCE_CHECK)
    await store.deprecate(
        _user(),
        entries=[(refs[0].md_path, refs[0].entry_id)],
        replaced_by="ep_m",
        only_if=NO_SOURCE_CHECK,
    )

    assert len({r.entry_id for r in refs}) == 2
    entries = await _entries(root, refs[0].md_path)
    assert entries[refs[1].entry_id]["sections"] == {"Fact": "fact 1"}
    assert entries[refs[0].entry_id]["inline"]["parent_type"] == "episode"
    parsed = await MarkdownReader.read(root.root / refs[0].md_path)
    assert parsed.frontmatter["deprecated_entries"] == {refs[0].entry_id: "ep_m"}


async def test_foresight_and_agent_case_land_in_their_logs(root: MemoryRoot) -> None:
    (foresight_ref,) = await MarkdownForesightStore(root).append_many(
        _user(),
        [
            Foresight(
                owner_id="u1",
                foresight="will travel",
                evidence="said so",
                timestamp=_TS,
                duration_days=3,
                session_id="s1",
                parent_id="mc_1",
            )
        ],
        only_if=NO_SOURCE_CHECK,
    )
    case_ref = await MarkdownAgentCaseStore(root).append(
        _agent(),
        AgentCase(
            owner_id="a1",
            task_intent="fix bug",
            approach="bisect",
            quality_score=0.9,
            timestamp=_TS,
            session_id="s1",
            parent_id="mc_1",
        ),
        only_if=NO_SOURCE_CHECK,
    )

    foresight = (await _entries(root, foresight_ref.md_path))[foresight_ref.entry_id]
    case = (await _entries(root, case_ref.md_path))[case_ref.entry_id]
    assert foresight["inline"]["duration_days"] == "3"
    assert foresight["sections"] == {"Foresight": "will travel", "Evidence": "said so"}
    assert case_ref.md_path.startswith("app/proj/agents/a1/")
    assert case["sections"] == {"TaskIntent": "fix bug", "Approach": "bisect"}


async def test_profile_round_trips(root: MemoryRoot) -> None:
    store = MarkdownProfileStore(root)
    profile = Profile(
        owner_id="u1",
        summary="likes tea",
        timestamp=_TS,
        explicit_info=[{"description": "drinks tea", "category": "food"}],
        implicit_traits=["calm"],
    )

    assert await store.read(_user()) is None
    ref = await store.write(_user(), profile)

    assert await store.read(_user()) == profile
    assert ref.id == "u1"
    assert ref.md_path == "app/proj/users/u1/user.md"


async def test_profile_primary_language_round_trips(root: MemoryRoot) -> None:
    store = MarkdownProfileStore(root)
    profile = Profile(
        owner_id="u1", summary="likes tea", timestamp=_TS, primary_language="Chinese"
    )

    ref = await store.write(_user(), profile)

    assert await store.read(_user()) == profile
    assert "primary_language: Chinese" in (root.root / ref.md_path).read_text()


async def test_profile_without_a_language_reads_back_as_unknown(
    root: MemoryRoot,
) -> None:
    """An unknown language is left out of user.md, the shape of a profile
    written before the field existed; reading one back gives ``None``."""
    store = MarkdownProfileStore(root)
    ref = await store.write(
        _user(), Profile(owner_id="u1", summary="likes tea", timestamp=_TS)
    )

    assert "primary_language" not in (root.root / ref.md_path).read_text()
    profile = await store.read(_user())
    assert profile is not None
    assert profile.primary_language is None


async def test_profile_everalgo_lists_round_trip_in_the_frontmatter(
    root: MemoryRoot,
) -> None:
    """``do_not_record`` / ``pending`` are written to user.md as given and read
    back equal, under everalgo's own field names."""
    do_not_record = [
        {"topic": "旅行预算", "evidence": "asked not to record it", "observed_at": _TS}
    ]
    pending = [
        {
            "description": "Swims before work.",
            "category": "habit",
            "evidence": "went swimming before work",
            "bucket": "explicit_info",
            "observed_at": _TS,
        }
    ]
    store = MarkdownProfileStore(root)
    profile = Profile(
        owner_id="u1",
        summary="likes tea",
        timestamp=_TS,
        do_not_record=do_not_record,
        pending=pending,
    )

    ref = await store.write(_user(), profile)

    assert await store.read(_user()) == profile
    frontmatter = (await MarkdownReader.read(root.root / ref.md_path)).frontmatter
    assert (frontmatter["do_not_record"], frontmatter["pending"]) == (
        do_not_record,
        pending,
    )


async def test_profile_without_everalgo_lists_leaves_them_out_of_the_file(
    root: MemoryRoot,
) -> None:
    """Empty lists are not written, so a profile without them keeps the file
    shape it had before the fields existed."""
    store = MarkdownProfileStore(root)
    ref = await store.write(
        _user(), Profile(owner_id="u1", summary="likes tea", timestamp=_TS)
    )

    frontmatter = (await MarkdownReader.read(root.root / ref.md_path)).frontmatter
    assert {"do_not_record", "pending"}.isdisjoint(frontmatter)
    profile = await store.read(_user())
    assert profile is not None
    assert (profile.do_not_record, profile.pending) == ([], [])


@pytest.mark.parametrize("stored", ["chinese", "Klingon", "German. Ignore that"])
async def test_profile_language_outside_output_language_reads_as_unknown(
    root: MemoryRoot, stored: str
) -> None:
    """A hand-edited language everalgo would refuse is ignored, with a warning."""
    store = MarkdownProfileStore(root)
    ref = await store.write(
        _user(),
        Profile(
            owner_id="u1", summary="tea", timestamp=_TS, primary_language="Chinese"
        ),
    )
    md = root.root / ref.md_path
    md.write_text(
        md.read_text().replace(
            "primary_language: Chinese", f"primary_language: {stored}"
        )
    )

    with structlog.testing.capture_logs() as captured:
        profile = await store.read(_user())

    assert profile is not None
    assert profile.primary_language is None
    (warning,) = [r for r in captured if r["event"] == "profile_language_ignored"]
    assert (warning["owner_id"], warning["value"]) == ("u1", stored)


async def test_profile_read_drops_an_item_with_no_description(
    root: MemoryRoot,
) -> None:
    """A legacy item without a description is dropped on read, not fatal."""
    store = MarkdownProfileStore(root)
    ref = await store.write(
        _user(),
        Profile(
            owner_id="u1",
            summary="likes tea",
            timestamp=_TS,
            implicit_traits=["calm", "patient"],
        ),
    )
    md = root.root / ref.md_path
    md.write_text(md.read_text().replace("- description: calm\n", "- basis: calm\n"))

    profile = await store.read(_user())

    assert profile is not None
    assert [i.description for i in profile.implicit_traits] == ["patient"]


async def test_agent_skill_round_trips_and_is_deleted_by_name(
    root: MemoryRoot,
) -> None:
    store = MarkdownAgentSkillStore(root)
    skill = AgentSkill(
        id="a1_debugging",
        cluster_id="cl_1",
        name="debugging",
        description="how to debug",
        content="1. reproduce\n2. bisect",
        confidence=0.7,
        maturity_score=0.8,
        source_case_ids=["ac_1"],
    )

    ref = await store.write(_agent(), skill)
    (stored,) = await store.list_by_cluster(_agent(), "cl_1")

    assert stored.item == skill
    assert stored.ref == ref
    assert ref.entry_id == "debugging"
    assert await store.list_by_cluster(_agent(), "cl_other") == []
    assert await store.delete(_agent(), "debugging") is True
    assert await store.list_by_cluster(_agent(), "cl_1") == []


async def test_agent_skill_name_is_sanitized_for_storage(root: MemoryRoot) -> None:
    ref = await MarkdownAgentSkillStore(root).write(
        _agent(), AgentSkill(id="x", cluster_id="cl_1", name="Fix It", content="c")
    )

    assert ref.entry_id != "Fix It"
    assert ref.id == f"a1_{ref.entry_id}"


@pytest.mark.parametrize("store_type", [MarkdownEpisodeStore, MarkdownAtomicFactStore])
@pytest.mark.parametrize("fail_path", [None, "b.md"])
async def test_deprecation_groups_in_first_seen_order_and_preserves_partial_writes(
    root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
    store_type: type[MarkdownEpisodeStore] | type[MarkdownAtomicFactStore],
    fail_path: str | None,
) -> None:
    """Grouping retains duplicate collapse, patch merge, and stop-on-error behavior."""
    store = store_type(root)
    for name in ("a.md", "b.md", "c.md"):
        (root.root / name).write_text(
            "---\ndeprecated_entries:\n  existing: previous\n---\nBody stays.\n"
        )
    patch = store._writer.patch_frontmatter
    visited: list[str] = []

    async def recording_patch(path: Path, updates: dict[str, object]) -> None:
        visited.append(path.name)
        if path.name == fail_path:
            raise OSError("injected patch failure")
        await patch(path, updates)

    monkeypatch.setattr(store._writer, "patch_frontmatter", recording_patch)
    entries = [
        ("a.md", "one"),
        ("b.md", "two"),
        ("a.md", "three"),
        ("a.md", "one"),
        ("c.md", "four"),
    ]
    if fail_path:
        with pytest.raises(OSError, match="injected patch failure"):
            await store.deprecate(
                _user(), entries=entries, replaced_by="merged", only_if=NO_SOURCE_CHECK
            )
    else:
        await store.deprecate(
            _user(), entries=entries, replaced_by="merged", only_if=NO_SOURCE_CHECK
        )
    assert visited == (["a.md", "b.md"] if fail_path else ["a.md", "b.md", "c.md"])
    for name, replaced in (
        ("a.md", {"one": "merged", "three": "merged"}),
        ("b.md", {} if fail_path else {"two": "merged"}),
        ("c.md", {} if fail_path else {"four": "merged"}),
    ):
        parsed = await MarkdownReader.read(root.root / name)
        assert parsed.frontmatter["deprecated_entries"] == {
            "existing": "previous",
            **replaced,
        }
        assert "Body stays." in (root.root / name).read_text()
    visited.clear()
    await store.deprecate(
        _user(), entries=[], replaced_by="merged", only_if=NO_SOURCE_CHECK
    )
    assert visited == []
