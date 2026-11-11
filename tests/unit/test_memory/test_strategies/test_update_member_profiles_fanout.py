"""update_member_profiles: a flush reads storage in a bounded fan-out.

docs/group-memory.md, section 7.4. A flush updates every member with pending
segments at once. Each member's update used to ask the memcell store for its
pending segments one id at a time, all at once, and every member started
together: GroupAssistBench probes logged hundreds of ``sqlite.pool.saturated``
warnings per flush. What these pin:

- one memcell read per member, holding all of that member's pending ids;
- never more than ``[sqlite] pool_size`` members reading storage at a time;
- the profiles written are the same whatever the bound, so the fix changes
  how the strategy reads and nothing it writes.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.config import load_settings
from everos.memory.events import GroupEpisodeExtracted, GroupFlushed
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import Scope
from everos.memory.strategies import update_member_profiles
from tests.fakes import make_deps, make_stores
from tests.fakes.stores import InMemoryMemcellStore

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_MEMBERS = [f"u_{n}" for n in range(6)]
_PENDING = 3


class _ObservedCells(InMemoryMemcellStore):
    """Counts the reads and how many of them are in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        self.keyed_calls: list[list[str]] = []
        self.single_calls = 0
        self.in_flight = 0
        self.peak = 0

    async def cells_by_id(self, memcell_ids: Sequence[str]) -> dict[str, MemCell]:
        self.keyed_calls.append(list(memcell_ids))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            for _ in range(5):  # let every other started read begin
                await asyncio.sleep(0)
            return await super().cells_by_id(memcell_ids)
        finally:
            self.in_flight -= 1

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]:
        self.single_calls += 1
        return await super().find_by_ids(memcell_ids)


async def _model(messages: Any, *, response_format: type, **_: Any) -> str:
    text = "\n".join(str(m.content) for m in messages)
    target = re.search(r"The target user is (.+?), the speaker", text)
    who = target.group(1).strip() if target else "?"
    item = {"evidence": f"{who} said so", "description": f"About {who}"}
    item["category"] = "plans"
    if response_format.__name__ == "_ProfileUpdateReply":
        op = {"action": "add", "type": "explicit_info", "data": item}
        return json.dumps({"operations": [op]})
    raise AssertionError(response_format.__name__)


def _without_ids(dumped: dict[str, Any]) -> dict[str, Any]:
    """The profile with each item's minted ``item_id`` left out (it is random)."""
    return {
        key: [{k: v for k, v in i.items() if k != "item_id"} for i in value]
        if key in ("explicit_info", "implicit_traits")
        else value
        for key, value in dumped.items()
    }


async def _flush_everyone(monkeypatch: pytest.MonkeyPatch, pool_size: int) -> Any:
    monkeypatch.setenv("EVEROS_SQLITE__POOL_SIZE", str(pool_size))
    monkeypatch.setenv("EVEROS_GROUPS__MEMBER_PROFILE_EVERY", "50")
    load_settings.cache_clear()
    cells = _ObservedCells()
    deps = make_deps(
        llm=FakeLLMClient(handler=_model), stores=make_stores(memcell=cells)
    )
    key = ("default", "p", "trip")
    deps.stores.groups.groups[key] = Group(group_id="trip", created_at=0)
    deps.stores.groups.events[key].extend(
        RosterEvent(op="join", member_id=m, at=0, display_name=m.upper())
        for m in _MEMBERS
    )
    for n, member in enumerate(_MEMBERS):
        for k in range(_PENDING):
            memcell_id = f"mc_{n}_{k}"
            at = 1_000 * (n * _PENDING + k + 1)
            item = ChatMessage(
                id=f"{memcell_id}_0",
                role="user",
                sender_id=member,
                sender_name=member,
                content=f"note {k} from {member}",
                timestamp=at,
            )
            cells.seed(memcell_id, MemCell(items=[item], timestamp=at + 1))
            await update_member_profiles(
                GroupEpisodeExtracted(
                    memcell_id=memcell_id,
                    episode_entry_id=f"gep_{memcell_id}",
                    episode_text="text",
                    episode_timestamp_ms=at,
                    group_id="trip",
                    app_id="default",
                    project_id="p",
                    speaker_ids=[member],
                    language="English",
                ),
                deps,
            )
    assert cells.keyed_calls == []  # all queued, below member_profile_every

    await update_member_profiles(
        GroupFlushed(group_id="trip", app_id="default", project_id="p", session_id="s"),
        deps,
    )
    return deps, cells


async def test_a_flush_reads_each_member_s_segments_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deps, cells = await _flush_everyone(monkeypatch, pool_size=5)

    assert cells.single_calls == 0
    assert sorted(cells.keyed_calls) == sorted(
        [f"mc_{n}_{k}" for k in range(_PENDING)] for n in range(len(_MEMBERS))
    )


@pytest.mark.parametrize("pool_size", [1, 2])
async def test_no_more_members_read_at_once_than_the_pool_holds(
    monkeypatch: pytest.MonkeyPatch, pool_size: int
) -> None:
    _deps, cells = await _flush_everyone(monkeypatch, pool_size=pool_size)

    assert len(cells.keyed_calls) == len(_MEMBERS)
    assert cells.peak == pool_size


async def test_the_bound_changes_nothing_that_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    narrow, _ = await _flush_everyone(monkeypatch, pool_size=1)
    wide, _ = await _flush_everyone(monkeypatch, pool_size=100)

    for member in _MEMBERS:
        one = await narrow.stores.member_profile.read(_SCOPE, member)
        other = await wide.stores.member_profile.read(_SCOPE, member)
        assert one is not None and other is not None
        assert one.pending_memcell_ids == other.pending_memcell_ids == []
        assert _without_ids(one.profile.model_dump()) == _without_ids(
            other.profile.model_dump()
        )
        sources = (one.profile.explicit_info[0].model_extra or {})["source_memcell_ids"]
        n = _MEMBERS.index(member)
        assert sources == [f"mc_{n}_{k}" for k in range(_PENDING)]
