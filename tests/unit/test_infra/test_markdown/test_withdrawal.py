"""Taking entries out of markdown and putting them back exactly (section 9.4).

A forget inside a group takes the previous versions of what it rewrites out
of the files -- what was forgotten must be nowhere under the memory root --
and its undo puts them back. Real markdown under a temporary root. What
these pin:

- a withdrawn entry leaves its file (its text is in the returned
  :class:`Withdrawal`, nowhere on disk) and a reinstatement puts back the
  same bytes, under the same id, in the same place among its neighbours;
- the file's ``entry_count`` never falls, so the next append never reuses a
  withdrawn id;
- an entry whose mark named a withdrawn one is current again (its mark is
  released), and the reinstatement marks it again; the withdrawn entry's own
  mark comes back with it;
- reinstating twice puts nothing back twice; a file that went missing is
  written again;
- the segment reads find a group's current episodes by segment and by the
  picture they refer to, and its picture memories by segment;
- removing a member's profile returns what it held and its index row.
"""

from __future__ import annotations

from pathlib import Path

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownEpisodeStore,
    MarkdownGroupProfileStore,
    MarkdownImageMemoryStore,
    MarkdownMemberProfileStore,
    MarkdownReader,
)
from everos.memory.models import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    MemberProfile,
    Profile,
    ProfileSource,
)
from everos.memory.ports import NO_SOURCE_CHECK, Scope

_GROUP = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_T0 = 1_772_442_120_000
_HASH = "ab" * 32


def _episode(text: str, memcell_id: str, image_ids: list[str] | None = None) -> Episode:
    return Episode(
        owner_id="trip",
        episode=text,
        summary=text[:20],
        subject=text[:10],
        timestamp=_T0,
        session_id="s1",
        sender_ids=["u_chen"],
        parent_id=memcell_id,
        source_start=_T0 - 1000,
        subject_ids=["u_chen"],
        origin="member",
        image_ids=image_ids or [],
    )


async def test_an_entry_goes_and_comes_back_byte_for_byte(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownEpisodeStore(root)
    refs = await store.append_many(
        _GROUP,
        [_episode(f"episode {n}", f"mc_{n}") for n in range(3)],
        only_if=NO_SOURCE_CHECK,
    )
    path = root.root / refs[1].md_path
    before = path.read_text("utf-8")

    withdrawal = await store.withdraw(_GROUP, [refs[1].entry_id])

    (entry,) = withdrawal.entries
    assert entry.entry_id == refs[1].entry_id and "episode 1" in entry.text
    assert "episode 1" not in path.read_text("utf-8")
    assert await store.read_many(_GROUP, [refs[1].entry_id]) == []
    later = await store.append(
        _GROUP, _episode("later", "mc_9"), only_if=NO_SOURCE_CHECK
    )
    assert later is not None and later.entry_id not in {r.entry_id for r in refs}

    put = await store.reinstate(_GROUP, withdrawal)
    assert [r.entry_id for r in put] == [refs[1].entry_id]
    entries = (await MarkdownReader.read(path)).entries
    assert [e.id for e in entries] == [
        refs[0].entry_id,
        refs[1].entry_id,
        refs[2].entry_id,
        later.entry_id,
    ]
    restored = path.read_text("utf-8")
    block = restored[restored.index(f"<!-- entry:{later.entry_id} -->") :]
    original = before[before.index("<!-- entry:") :]
    assert restored[restored.index("<!-- entry:") :].replace(block, "") == original
    assert await store.reinstate(_GROUP, withdrawal) == []  # never twice


async def test_a_withdrawn_successor_releases_the_version_it_replaced(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownGroupProfileStore(root)

    def item(statement: str, supersedes: str | None = None) -> GroupProfileItem:
        return GroupProfileItem(
            owner_id="trip",
            kind="decision",
            statement=statement,
            sources=[
                ProfileSource(
                    memcell_id="mc_1",
                    message_id="m-1",
                    speaker_id="u_chen",
                    at=_T0,
                    quote=statement,
                )
            ],
            source_start=_T0,
            source_end=_T0,
            observed_at=_T0,
            supersedes=supersedes,
        )

    (old,) = await store.append(_GROUP, [item("fly May 1")], only_if=NO_SOURCE_CHECK)
    (new,) = await store.append(
        _GROUP, [item("fly April 30", old.entry_id)], only_if=NO_SOURCE_CHECK
    )
    (newest,) = await store.append(
        _GROUP, [item("fly April 29", new.entry_id)], only_if=NO_SOURCE_CHECK
    )

    withdrawal = await store.withdraw(_GROUP, [new.entry_id])

    (taken,) = withdrawal.entries
    assert taken.mark == newest.entry_id  # its own mark goes with it
    assert withdrawal.released == ((old.md_path, old.entry_id, new.entry_id),)
    statuses = {s.ref.entry_id: s.item.status for s in await store.items(_GROUP)}
    assert statuses == {old.entry_id: "current", newest.entry_id: "current"}

    await store.reinstate(_GROUP, withdrawal)
    after = {s.ref.entry_id: s.item for s in await store.items(_GROUP)}
    assert after[old.entry_id].superseded_by == new.entry_id
    assert after[new.entry_id].superseded_by == newest.entry_id
    assert after[newest.entry_id].status == "current"


async def test_a_missing_file_is_written_again(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownImageMemoryStore(root)
    ref = await store.append(
        _GROUP,
        ImageMemory(
            owner_id="trip",
            description="the venue",
            timestamp=_T0,
            image_id=_HASH,
            session_id="s1",
            parent_id="mc_1",
            sender_ids=["u_chen"],
        ),
        only_if=NO_SOURCE_CHECK,
    )
    assert ref is not None
    withdrawal = await store.withdraw(_GROUP, [ref.entry_id])
    (root.root / ref.md_path).unlink()

    await store.reinstate(_GROUP, withdrawal)

    (back,) = await store.read_many(_GROUP, [ref.entry_id])
    assert back.item.description == "the venue" and back.item.status == "current"
    (by_segment,) = await store.read_for_segments(_GROUP, ["mc_1"])
    assert by_segment.ref.entry_id == ref.entry_id


async def test_segment_reads_find_current_episodes(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownEpisodeStore(root)
    first, second, other = await store.append_many(
        _GROUP,
        [
            _episode(f"refers to [IMAGE: {_HASH}]", "mc_1", [_HASH]),
            _episode("plain", "mc_1"),
            _episode("elsewhere", "mc_10"),
        ],
        only_if=NO_SOURCE_CHECK,
    )
    await store.deprecate(
        _GROUP,
        entries=[(second.md_path, second.entry_id)],
        replaced_by=other.entry_id,
        only_if=NO_SOURCE_CHECK,
    )

    found = await store.read_for_segments(_GROUP, ["mc_1"])
    assert [s.ref.entry_id for s in found] == [first.entry_id]  # mc_10 is not mc_1
    referring = await store.read_referring(_GROUP, _HASH)
    assert [s.ref.entry_id for s in referring] == [first.entry_id]


async def test_a_member_profile_is_removed_with_its_row(tmp_path: Path) -> None:
    store = MarkdownMemberProfileStore(MemoryRoot(tmp_path))
    profile = MemberProfile(
        group_id="trip",
        member_id="u_chen",
        profile=Profile(owner_id="u_chen", summary="likes trains", timestamp=_T0),
    )
    await store.write(_GROUP, profile)

    removed = await store.remove(_GROUP, "u_chen")

    assert removed is not None
    assert removed.item.profile.summary == "likes trains"
    assert removed.ref.id == "trip:u_chen"
    assert await store.read(_GROUP, "u_chen") is None
    assert await store.remove(_GROUP, "u_chen") is None
