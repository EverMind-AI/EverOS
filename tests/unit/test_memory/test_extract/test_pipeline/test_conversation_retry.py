"""``ConversationPipeline.run_cell`` on a retried cell (``PreparedCell.retry``).

An earlier extraction of the cell failed, possibly after writing some of its
owners' copies: an owner whose episode of it exists is not written again, and
a cell every owner of which has one makes no model call at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.extract.pipeline import ConversationPipeline, PreparedCell
from everos.memory.ports import MemoryRef
from tests.fakes import bundled_instructions, user_track_llm
from tests.fakes.stores import InMemoryGroupStore, InMemoryMemcellStore

_REF = MemoryRef(
    id="u2_ep_20260517_00000001",
    entry_id="ep_20260517_00000001",
    md_path="default_app/default_project/users/u2/episodes/episode-2026-05-17.md",
)
_INGESTED = IngestResult(session_id="s1", messages=[])


def _cell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"m_{sender}",
                role="user",
                content="We fly to Porto on Friday",
                timestamp=1_700_000_000_000,
                sender_id=sender,
            )
            for sender in ("u1", "u2")
        ],
        timestamp=1_700_000_000_000,
    )


def _pipeline(llm: object, written: set[str]) -> tuple[ConversationPipeline, MagicMock]:
    episodes = MagicMock()
    episodes.append = AsyncMock(return_value=_REF)

    async def read_for_segments(scope: object, memcell_ids: object) -> list[object]:
        return [object()] if scope.owner_id in written else []  # type: ignore[attr-defined]

    episodes.read_for_segments = AsyncMock(side_effect=read_for_segments)
    memcells = InMemoryMemcellStore()
    memcells.seed("mc_a", _cell())
    pipeline = ConversationPipeline(
        episode_store=episodes,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
        groups=InMemoryGroupStore(),
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient
    )
    return pipeline, episodes


async def test_a_retry_writes_only_the_copies_that_are_missing() -> None:
    llm = user_track_llm()
    pipeline, episodes = _pipeline(llm, written={"u1"})
    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(
            cell=_cell(), memcell_id="mc_a", sender_ids=["u1", "u2"], retry=True
        ),
    )
    assert llm.call_count == 1
    assert [call.args[0].owner_id for call in episodes.append.await_args_list] == ["u2"]
    assert [event.owner_id for event in outcome.events] == ["u2"]


async def test_a_retry_whose_copies_all_exist_makes_no_call() -> None:
    llm = user_track_llm()
    pipeline, episodes = _pipeline(llm, written={"u1", "u2"})
    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(
            cell=_cell(), memcell_id="mc_a", sender_ids=["u1", "u2"], retry=True
        ),
    )
    assert llm.call_count == 0
    episodes.append.assert_not_awaited()
    assert (outcome.message_count, list(outcome.events)) == (2, [])


async def test_a_first_extraction_reads_no_existing_copy() -> None:
    llm = user_track_llm()
    pipeline, episodes = _pipeline(llm, written={"u1", "u2"})
    await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_cell(), memcell_id="mc_a", sender_ids=["u1", "u2"]),
    )
    episodes.read_for_segments.assert_not_awaited()
    assert episodes.append.await_count == 2
