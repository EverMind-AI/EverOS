"""The group profile's markdown store (docs/group-memory.md, sections 6.1 and 7.5).

Real markdown under a temporary root. What these pin:

- items round-trip through ``groups/<g>/profile/group_profile-<date>.md`` with
  their verified evidence -- a quote holding quotes, colons, CJK and a long
  line comes back verbatim -- and come back with ``item_id`` and their own
  status;
- the entry is the documented format: inline fields only when set, the times
  in UTC, ``sources`` informational, one JSON line per source under
  ``Evidence``;
- a new version marks the entry it replaces in that entry's own file, in the
  same append, across two daily files too; ``items`` reads ``superseded`` and
  ``superseded_by`` back, oldest file first;
- a forget marks an entry ``forgotten``, and a later version naming it never
  turns it back into ``superseded``;
- a version of an entry that is not there is written and marks nothing;
- nothing is written once ``only_if`` finds the sources gone;
- the pending batch round-trips in ``profile/pending.md``, which is no daily
  log and which no cascade kind indexes;
- a hand-broken entry is skipped, the rest stays readable;
- any scope but a group's is refused.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from everos.core.utils.datetime import today_with_timezone
from everos.infra.cascade.registry import KIND_REGISTRY, match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    GroupProfileWriter,
    MarkdownGroupProfileStore,
    MarkdownReader,
    render_group_profile_entry,
)
from everos.memory.models import FORGOTTEN_MARK, GroupProfileItem, ProfileSource
from everos.memory.ports import GroupProfileStore, NoSourceCheck, PendingBatch, Scope

_GROUP = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_USER = Scope(app_id="default", project_id="p", owner_id="u_chen", owner_type="user")
_PROFILE_DIR = "default_app/p/groups/trip/profile"
_T0 = 1_772_442_120_000  # 2026-03-02T09:02:00Z
_READ_SIDE = {"item_id", "status", "superseded_by"}
_QUOTE = (
    'Chen: "Let\'s leave on April 30" -- 好的, flight at 08:10: book it; '
    + "the long tail of a long message " * 20
)


def _source(message_id: str, speaker: str, at: int, quote: str) -> ProfileSource:
    return ProfileSource(
        memcell_id="mc_7f3a",
        message_id=message_id,
        speaker_id=speaker,
        at=at,
        quote=quote,
    )


def _item(kind: str, **fields: Any) -> GroupProfileItem:
    values: dict[str, Any] = {
        "owner_id": "trip",
        "kind": kind,
        "sources": [
            _source("m-102", "u_chen", _T0, _QUOTE),
            _source("m-103", "u_betty", _T0 + 180_000, "Fine by me."),
        ],
        "subject_ids": ["u_chen", "u_betty"],
        "sender_ids": ["u_chen", "u_betty"],
        "origin": "member",
        "source_start": _T0,
        "source_end": _T0 + 180_000,
        "observed_at": _T0 + 360_000,
    }
    values.update(fields)
    return GroupProfileItem(**values)


def _decision(**fields: Any) -> GroupProfileItem:
    values: dict[str, Any] = {
        "statement": "Depart on April 30 instead of May 1.",
        "topic_id": "tp_4c1e",
        "decided_by": ["u_chen"],
        "agreed_by": ["u_betty"],
        "assignees": ["u_chen"],
    }
    values.update(fields)
    return _item("decision", **values)


def _topic() -> GroupProfileItem:
    return _item(
        "topic",
        topic_id="tp_4c1e",
        name="Departure date",
        summary="Moved from May 1 to April 30: the May 1 flights sold out.",
        topic_status="consensus",
    )


def _dissent() -> GroupProfileItem:
    return _item(
        "dissent",
        statement="Betty would rather leave on May 2.",
        held_by=["u_betty"],
        about="tp_4c1e",
        resolved=True,
        resolved_by=["u_chen"],
    )


def _role() -> GroupProfileItem:
    return _item(
        "role",
        member_id="u_chen",
        role="decision_maker",
        confidence="strong",
        retired=True,
        applies_to=["u_chen"],
    )


def _today_file() -> str:
    return f"{_PROFILE_DIR}/group_profile-{today_with_timezone().isoformat()}.md"


async def test_items_round_trip_with_their_evidence(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    assert isinstance(store, GroupProfileStore)
    written = [_topic(), _decision(), _dissent(), _role()]

    refs = await store.append(_GROUP, written, only_if=NoSourceCheck())

    day = today_with_timezone()
    assert [ref.md_path for ref in refs] == [_today_file()] * 4
    assert [ref.entry_id for ref in refs] == [
        f"gp_{day:%Y%m%d}_{seq:08d}" for seq in (1, 2, 3, 4)
    ]
    assert [ref.id for ref in refs] == [f"trip_{ref.entry_id}" for ref in refs]
    stored = await store.items(_GROUP)
    assert [s.ref for s in stored] == refs
    assert all(s.scope == _GROUP for s in stored)
    assert [s.item.model_dump(exclude=_READ_SIDE) for s in stored] == [
        item.model_dump(exclude=_READ_SIDE) for item in written
    ]
    assert stored[1].item.sources[0].quote == _QUOTE
    assert [s.item.item_id for s in stored] == [ref.entry_id for ref in refs]
    assert [s.item.status for s in stored] == [
        "consensus",
        "current",
        "resolved",
        "retired",
    ]
    assert [s.item.superseded_by for s in stored] == [None] * 4


async def test_a_quote_with_line_breaks_stays_on_its_evidence_line(
    tmp_path: Path,
) -> None:
    quote = "first line\nsecond\u2028third\u2029fourth\x85fifth\r\nend"
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    item = _decision(sources=[_source("m-1", "u_chen", _T0, quote)])
    await store.append(_GROUP, [item], only_if=NoSourceCheck())

    (stored,) = await store.items(_GROUP)
    assert stored.item.sources[0].quote == quote


async def test_the_entry_is_the_documented_format(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    (ref,) = await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())

    text = (tmp_path / ref.md_path).read_text(encoding="utf-8")
    entry = text.split(f"<!-- entry:{ref.entry_id} -->\n", 1)[1]
    entry = entry.split(f"<!-- /entry:{ref.entry_id} -->", 1)[0]
    assert [line for line in entry.splitlines() if line.startswith("**")] == [
        "**kind**: decision",
        "**topic_id**: tp_4c1e",
        "**decided_by**: [u_chen]",
        "**agreed_by**: [u_betty]",
        "**assignees**: [u_chen]",
        "**subject_ids**: [u_chen, u_betty]",
        "**sender_ids**: [u_chen, u_betty]",
        "**origin**: member",
        "**sources**: [mc_7f3a/m-102, mc_7f3a/m-103]",
        "**source_start**: 2026-03-02T09:02:00+00:00",
        "**source_end**: 2026-03-02T09:05:00+00:00",
        "**observed_at**: 2026-03-02T09:08:00+00:00",
    ]
    assert "### Statement\nDepart on April 30 instead of May 1.\n" in entry
    assert "### Name" not in entry and "### Summary" not in entry
    evidence = entry.split("### Evidence\n", 1)[1].splitlines()
    assert evidence == [
        "- " + json.dumps(source.model_dump(), ensure_ascii=False)
        for source in _decision().sources
    ]
    assert evidence[1] == (
        '- {"memcell_id": "mc_7f3a", "message_id": "m-103", '
        '"speaker_id": "u_betty", "at": 1772442300000, "quote": "Fine by me."}'
    )


async def test_flags_are_written_only_when_true(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    await store.append(
        _GROUP,
        [_dissent(), _decision(statement="Hotel by the station")],
        only_if=NoSourceCheck(),
    )

    text = (tmp_path / _today_file()).read_text(encoding="utf-8")
    assert text.count("**resolved**: true") == 1  # the dissent's, not the decision's
    assert "**resolved_by**: [u_chen]" in text
    assert "**retired**" not in text


async def test_a_new_version_marks_the_one_it_replaces_in_that_ones_file(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownGroupProfileStore(root)
    earlier = today_with_timezone() - dt.timedelta(days=3)
    (old,) = await GroupProfileWriter(root).append_entries(
        "trip",
        [render_group_profile_entry(_decision())],
        date=earlier,
        app_id="default",
        project_id="p",
    )
    old_id = old.format()

    (new,) = await store.append(
        _GROUP,
        [_decision(statement="Depart on May 2.", supersedes=old_id)],
        only_if=NoSourceCheck(),
    )

    old_file = tmp_path / _PROFILE_DIR / f"group_profile-{earlier.isoformat()}.md"
    marks = (await MarkdownReader.read(old_file)).frontmatter["deprecated_entries"]
    assert marks == {old_id: new.entry_id}
    new_file = (await MarkdownReader.read(tmp_path / new.md_path)).frontmatter
    assert "deprecated_entries" not in new_file
    stored = await store.items(_GROUP)
    assert [(s.ref.entry_id, s.item.status, s.item.superseded_by) for s in stored] == [
        (old_id, "superseded", new.entry_id),
        (new.entry_id, "current", None),
    ]
    assert stored[1].item.supersedes == old_id


async def test_a_forgotten_entry_stays_forgotten(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    (first,) = await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())

    assert await store.deprecate(
        _GROUP,
        entries=[(first.md_path, first.entry_id)],
        replaced_by=FORGOTTEN_MARK,
        only_if=NoSourceCheck(),
    )
    (later,) = await store.append(
        _GROUP,
        [_decision(statement="Depart on May 2.", supersedes=first.entry_id)],
        only_if=NoSourceCheck(),
    )

    stored = await store.items(_GROUP)
    assert [(s.ref.entry_id, s.item.status, s.item.superseded_by) for s in stored] == [
        (first.entry_id, "forgotten", None),
        (later.entry_id, "current", None),
    ]


async def test_a_version_of_an_entry_that_is_not_there_marks_nothing(
    tmp_path: Path,
) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    (present,) = await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())

    refs = await store.append(
        _GROUP,
        [
            _decision(supersedes="gp_20200101_00000009"),  # no such file
            _decision(supersedes=f"{present.entry_id[:-1]}7"),  # no such entry
            _decision(supersedes="not-an-entry-id"),
        ],
        only_if=NoSourceCheck(),
    )

    assert len(refs) == 3
    stored = await store.items(_GROUP)
    assert [s.item.status for s in stored] == ["current"] * 4
    assert sorted(p.name for p in (tmp_path / _PROFILE_DIR).iterdir()) == [
        Path(_today_file()).name
    ]


async def test_nothing_is_written_once_the_sources_are_gone(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))

    async def gone() -> bool:
        return False

    assert await store.append(_GROUP, [_decision()], only_if=gone) == []
    assert not await store.deprecate(
        _GROUP,
        entries=[(_today_file(), "gp_20260302_00000001")],
        replaced_by=FORGOTTEN_MARK,
        only_if=gone,
    )
    assert await store.items(_GROUP) == []
    assert not (tmp_path / "default_app").exists()


async def test_the_pending_batch_round_trips_beside_the_logs(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    assert await store.pending(_GROUP) == []

    await store.set_pending(_GROUP, ["mc_1", "mc_2"])
    await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())

    assert await store.pending(_GROUP) == ["mc_1", "mc_2"]
    parsed = await MarkdownReader.read(tmp_path / _PROFILE_DIR / "pending.md")
    assert parsed.entries == []
    assert parsed.frontmatter["type"] == "group_profile_pending"
    assert (parsed.frontmatter["group_id"], parsed.frontmatter["track"]) == (
        "trip",
        "group",
    )
    assert parsed.frontmatter["pending_memcell_ids"] == ["mc_1", "mc_2"]
    assert len(await store.items(_GROUP)) == 1  # pending.md is no daily log
    await store.set_pending(_GROUP, [])
    assert await store.pending(_GROUP) == []


async def test_the_batch_keeps_its_failed_updates(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    assert await store.pending_batch(_GROUP) == PendingBatch()

    await store.set_pending(_GROUP, ["mc_1"], failures=2, retry_at=1_234)

    assert await store.pending_batch(_GROUP) == PendingBatch(
        memcell_ids=("mc_1",), failures=2, retry_at=1_234
    )
    parsed = await MarkdownReader.read(tmp_path / _PROFILE_DIR / "pending.md")
    assert (parsed.frontmatter["failures"], parsed.frontmatter["retry_at"]) == (
        2,
        1_234,
    )
    # A plain rewrite of the queue clears the record.
    await store.set_pending(_GROUP, ["mc_1", "mc_2"])
    assert await store.pending_batch(_GROUP) == PendingBatch(
        memcell_ids=("mc_1", "mc_2")
    )


async def test_a_pending_file_from_before_the_record_reads_as_none(
    tmp_path: Path,
) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    await store.set_pending(_GROUP, ["mc_1"])
    path = tmp_path / _PROFILE_DIR / "pending.md"
    text = path.read_text(encoding="utf-8")
    old = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("failures:", "retry_at:"))
    )
    path.write_text(old + "\n", encoding="utf-8")

    assert await store.pending_batch(_GROUP) == PendingBatch(memcell_ids=("mc_1",))


async def test_no_cascade_kind_indexes_the_pending_batch(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))
    await store.set_pending(_GROUP, ["mc_1"])
    (ref,) = await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())

    assert match_kind(f"{_PROFILE_DIR}/pending.md") is None
    spec = match_kind(ref.md_path)
    assert spec is not None and spec.name == "group_profile"
    assert _scanned(tmp_path) == {ref.md_path}


def _scanned(root: Path) -> set[str]:
    """What the cascade scanner walks: every kind's glob, from the root."""
    return {
        path.relative_to(root).as_posix()
        for kind in KIND_REGISTRY
        for path in root.glob(kind.path_glob())
    }


async def test_a_hand_broken_entry_is_skipped(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownGroupProfileStore(root)
    (good,) = await store.append(_GROUP, [_decision()], only_if=NoSourceCheck())
    await GroupProfileWriter(root).append_entries(
        "trip",
        [({"kind": "decision"}, {"Statement": "No times at all"})],
        app_id="default",
        project_id="p",
    )

    assert [s.ref.entry_id for s in await store.items(_GROUP)] == [good.entry_id]


async def test_a_group_profile_needs_a_group_scope(tmp_path: Path) -> None:
    store = MarkdownGroupProfileStore(MemoryRoot(tmp_path))

    with pytest.raises(ValueError, match="group scope"):
        await store.append(_USER, [_decision()], only_if=NoSourceCheck())
    with pytest.raises(ValueError, match="group scope"):
        await store.items(_USER)
    with pytest.raises(ValueError, match="group scope"):
        await store.deprecate(
            _USER, entries=[], replaced_by=FORGOTTEN_MARK, only_if=NoSourceCheck()
        )
    with pytest.raises(ValueError, match="group scope"):
        await store.pending(_USER)
    with pytest.raises(ValueError, match="group scope"):
        await store.set_pending(_USER, ["mc_1"])
    assert not (tmp_path / "default_app").exists()
