"""The profile cues' storage: ``profile_cue``, one row per ``(segment, person)``.

What these pin:

- a segment's cues are written once per person and read back per person,
  an empty record included (it tells "nothing marked" from "never marked");
- a record written again (a retried extraction) takes the new marks and
  keeps whether it was read; ``mark_read`` stamps only the person's records;
- a session's purge deletes its cue rows with its memcells.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from everalgo.types import ChatMessage, MemCell

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteDatabase,
    SqliteMemcellStore,
    SqliteProfileCueStore,
    SqliteRepos,
    SqliteSessionPurgeStore,
)
from everos.memory.models import MemcellRecord
from everos.memory.ports import PersonCues, SessionKey

_S1 = SessionKey(app_id="app", project_id="proj", session_id="s1")
_CAT = "c" * 64


@pytest.fixture
async def repos(tmp_path: Path) -> AsyncIterator[SqliteRepos]:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    await database.create_schema()
    yield SqliteRepos.over(database)
    await database.dispose()


async def test_a_segments_cues_are_read_back_per_person(repos: SqliteRepos) -> None:
    store = SqliteProfileCueStore(repos.profile_cue)
    await store.record(
        _S1,
        "mc_1",
        {
            "u_ann": PersonCues(message_ids=("m1", "m3"), image_ids=(_CAT,)),
            "u_bob": PersonCues(),
        },
    )

    assert await store.of_person("u_ann", ["mc_1", "mc_2"]) == {
        "mc_1": PersonCues(message_ids=("m1", "m3"), image_ids=(_CAT,))
    }
    assert await store.of_person("u_bob", ["mc_1"]) == {"mc_1": PersonCues()}
    assert await store.of_person("u_cat", ["mc_1"]) == {}


async def test_a_record_written_again_keeps_whether_it_was_read(
    repos: SqliteRepos,
) -> None:
    store = SqliteProfileCueStore(repos.profile_cue)
    await store.record(_S1, "mc_1", {"u_ann": PersonCues(message_ids=("m1",))})
    await store.record(_S1, "mc_1", {"u_bob": PersonCues()})
    await store.mark_read("u_ann", ["mc_1", "mc_gone"])

    await store.record(_S1, "mc_1", {"u_ann": PersonCues(message_ids=("m2",))})

    assert await store.of_person("u_ann", ["mc_1"]) == {
        "mc_1": PersonCues(message_ids=("m2",), read=True)
    }
    assert await store.of_person("u_bob", ["mc_1"]) == {"mc_1": PersonCues()}


async def test_a_sessions_purge_takes_its_cues(repos: SqliteRepos) -> None:
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hi", sender_id="u_ann", timestamp=1
            )
        ],
        timestamp=1,
    )
    await SqliteMemcellStore(repos.memcell).insert_many(
        _S1,
        [
            MemcellRecord(
                memcell_id="mc_1",
                cell=cell,
                raw_type="conversation",
                message_ids=["m1"],
                sender_ids=["u_ann"],
            )
        ],
    )
    store = SqliteProfileCueStore(repos.profile_cue)
    await store.record(_S1, "mc_1", {"u_ann": PersonCues(message_ids=("m1",))})

    await SqliteSessionPurgeStore(repos.session_purge).purge(_S1)

    assert await store.of_person("u_ann", ["mc_1"]) == {}
