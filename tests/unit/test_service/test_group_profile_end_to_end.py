"""The group profile over real markdown, system.db and LanceDB (task 11).

docs/group-memory.md, section 7.5 and task 11. The update strategy runs over
segments stored as the boundary stores them, with a scripted model; the
cascade projects ``groups/<g>/profile/`` into ``group_profile_item``; reads go
through ``/memory/get``. What these pin:

- superseding a decision appends the new item; the old one indexes as
  superseded; a current read lists the new one only and a history read
  (``include_superseded``) lists both;
- a read holds only items whose sources lie in the asker's windows;
- the bootstrap registers the idle trigger with ``[groups]
  group_profile_idle_seconds`` and turns both strategies off with the switch.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.config import Settings, load_settings
from everos.memory.events import GroupEpisodeExtracted
from everos.memory.models import MemcellRecord
from everos.memory.ports import Idle
from everos.memory.strategies import (
    UPDATE_GROUP_PROFILE_IDLE_SPEC,
    UPDATE_GROUP_PROFILE_SPEC,
    update_group_profile,
)
from everos.service import GetRequest
from everos.service.bootstrap import configured_spec
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000

_BLANK: dict[str, Any] = {
    "target": "",
    "topic_id": "",
    "name": "",
    "summary": "",
    "status": "",
    "statement": "",
    "decided_by": [],
    "agreed_by": [],
    "assignees": [],
    "applies_to": [],
    "held_by": [],
    "about": "",
    "member_id": "",
    "role": "",
    "confidence": "",
}


class _Model:
    def __init__(self) -> None:
        self.replies: list[list[dict[str, Any]]] = []

    def __call__(self, messages: Any, **_: Any) -> str:
        return json.dumps({"operations": self.replies.pop(0) if self.replies else []})


@pytest.fixture
async def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[GroupWorld, _Model]]:
    monkeypatch.setenv("EVEROS_GROUPS__GROUP_PROFILE_BATCH", "1")
    load_settings.cache_clear()
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as w:
        await w.roster(
            "trip",
            [
                join("u_chen", T0, role="owner"),
                join("u_betty", T0),
                join("u_evan", T0 + 5000),
            ],
        )
        yield w, model


async def _segment(
    world: GroupWorld, memcell_id: str, *lines: tuple[str, str, str, int]
) -> None:
    items = [
        ChatMessage(
            id=mid,
            role="user",
            sender_id=who,
            sender_name=who,
            content=text,
            timestamp=at,
        )
        for mid, who, text, at in lines
    ]
    cell = MemCell(items=items, timestamp=max(at for *_, at in lines))
    await world.stores.memcell.insert_many(
        world.session("s_trip"),
        [
            MemcellRecord(
                memcell_id=memcell_id,
                cell=cell,
                raw_type="Conversation",
                message_ids=[mid for mid, *_ in lines],
                sender_ids=list(dict.fromkeys(who for _, who, _, _ in lines)),
            )
        ],
    )
    await update_group_profile(
        GroupEpisodeExtracted(
            memcell_id=memcell_id,
            episode_entry_id=f"gep_{memcell_id}",
            episode_text="text",
            episode_timestamp_ms=cell.timestamp,
            group_id="trip",
            app_id=APP,
            project_id=PROJECT,
            speaker_ids=["u_chen"],
        ),
        world.deps,
    )


def _decision(evidence: list[tuple[str, str, str]], **fields: Any) -> dict[str, Any]:
    return {
        **_BLANK,
        **fields,
        "op": fields.pop("op", "decision.new"),
        "evidence": [{"segment": s, "message": m, "quote": q} for s, m, q in evidence],
    }


async def _listed(world: GroupWorld, asker: str, *, history: bool = False) -> list[Any]:
    response = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="group_profile",
            include_superseded=history,
        )
    )
    return response.data.group_profile


async def test_a_superseded_decision_reads_through_history(
    world: tuple[GroupWorld, _Model],
) -> None:
    w, model = world
    model.replies.append(
        [
            _decision(
                [("mc_1", "m-1", "fly out on May 1")],
                statement="Fly out on May 1.",
                decided_by=["u_chen"],
            )
        ]
    )
    await _segment(w, "mc_1", ("m-1", "u_chen", "We fly out on May 1.", T0 + 1000))
    (first,) = [s.item for s in await w.stores.group_profile.items(w.scope("trip"))]
    model.replies.append(
        [
            {
                **_decision(
                    [("mc_2", "m-2", "Let's leave on April 30 instead")],
                    statement="Depart on April 30 instead of May 1.",
                    decided_by=["u_chen"],
                ),
                "op": "decision.supersede",
                "target": first.item_id,
            }
        ]
    )
    await _segment(
        w,
        "mc_2",
        ("m-2", "u_chen", "Sold out. Let's leave on April 30 instead.", T0 + 6000),
    )
    await w.project()

    current = await _listed(w, "u_chen")
    history = await _listed(w, "u_chen", history=True)

    assert [(i.statement, i.status) for i in current] == [
        ("Depart on April 30 instead of May 1.", "current")
    ]
    assert [(i.statement, i.status) for i in history] == [
        ("Depart on April 30 instead of May 1.", "current"),
        ("Fly out on May 1.", "superseded"),
    ]
    old, new = history[1], history[0]
    assert old.superseded_by == new.item_id and new.supersedes == old.item_id
    assert new.sources[0].quote == "Let's leave on April 30 instead"
    assert new.sources[0].speaker_id == "u_chen"
    # Evan joined at T0 + 5000: the new decision is his to see, the old is not.
    assert [i.statement for i in await _listed(w, "u_evan", history=True)] == [
        "Depart on April 30 instead of May 1."
    ]


def test_the_bootstrap_registers_the_configured_triggers() -> None:
    base = Settings()
    settings = base.model_copy(
        update={
            "groups": base.groups.model_copy(update={"group_profile_idle_seconds": 90})
        }
    )
    idle = configured_spec(UPDATE_GROUP_PROFILE_IDLE_SPEC, settings)
    assert isinstance(idle.trigger, Idle)
    assert (idle.trigger.idle_seconds, idle.trigger.scan_interval_seconds) == (90, 45)
    assert idle.trigger.event_field == "group_key" and idle.enabled

    off = base.model_copy(
        update={"groups": base.groups.model_copy(update={"group_profile": False})}
    )
    assert not configured_spec(UPDATE_GROUP_PROFILE_IDLE_SPEC, off).enabled
    assert not configured_spec(UPDATE_GROUP_PROFILE_SPEC, off).enabled
    assert configured_spec(UPDATE_GROUP_PROFILE_SPEC, base).enabled
