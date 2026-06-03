from __future__ import annotations

import datetime as _dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Episode as AlgoEpisode

from everos.core.persistence import EntryId
from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.pipeline.user_memory import UserMemoryPipeline
from everos.memory.models import CanonicalMessage


def _sample_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hello",
                timestamp=1_700_000_000_000,
                sender_id="u1",
            ),
        ],
        timestamp=1_700_000_000_000,
    )


class _CapturingEngine:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    async def emit(self, event: object) -> None:
        self.emitted.append(event)


async def test_emit_pipeline_started_routes_through_engine() -> None:
    engine = _CapturingEngine()
    pipeline = UserMemoryPipeline(
        episode_writer=MagicMock(),
        prompt_loader=MagicMock(),
        llm_client=MagicMock(),
        engine=engine,
    )

    cell = _sample_memcell()
    await pipeline._emit_pipeline_started(  # noqa: SLF001 — test introspection
        memcell_id="mc_a",
        session_id="s1",
        app_id="claude_code",
        project_id="oss",
        cell=cell,
    )

    started = [e for e in engine.emitted if isinstance(e, UserPipelineStarted)]
    assert len(started) == 1
    assert started[0].memcell_id == "mc_a"
    assert started[0].session_id == "s1"
    assert started[0].app_id == "claude_code"
    assert started[0].project_id == "oss"
    assert started[0].memcell is cell


@pytest.mark.asyncio
async def test_emit_episode_extracted_after_md_write() -> None:
    """Each per-sender Episode write emits EpisodeExtracted with the md entry id."""
    engine = _CapturingEngine()
    episode_writer = MagicMock()
    episode_writer.append_entry = AsyncMock(
        return_value=EntryId(prefix="ep", date=_dt.date(2026, 5, 17), seq=1)
    )
    episode_writer.path_for = MagicMock(
        return_value="users/u1/episodes/episode-2026-05-17.md"
    )
    prompt_loader = MagicMock()
    prompt_loader.load = MagicMock(return_value="<prompt>")
    llm_client = MagicMock()

    pipeline = UserMemoryPipeline(
        episode_writer=episode_writer,
        prompt_loader=prompt_loader,
        llm_client=llm_client,
        engine=engine,
    )

    cell = _sample_memcell()
    ingested = IngestResult(
        session_id="s1",
        messages=[
            CanonicalMessage(
                message_id="m1",
                session_id="s1",
                sender_id="u1",
                role="user",
                timestamp=_dt.datetime.fromtimestamp(1_700_000_000, tz=_dt.UTC),
                text="hello",
            )
        ],
    )
    algo_ep = AlgoEpisode(
        owner_id="u1", episode="they said hello", timestamp=1_700_000_000_000
    )
    with patch.object(  # noqa: SLF001
        pipeline._ep_ext, "aextract", new=AsyncMock(return_value=algo_ep)
    ):
        outcome = await pipeline.run(
            ingested=ingested,
            cells=[cell],
            memcell_ids=["mc_a"],
            per_cell_all_senders=[["u1"]],
        )

    assert outcome.status == "extracted"
    extracted = [e for e in engine.emitted if isinstance(e, EpisodeExtracted)]
    assert len(extracted) == 1
    assert extracted[0].memcell_id == "mc_a"
    assert extracted[0].episode_entry_id == "ep_20260517_00000001"
    assert extracted[0].episode_text == "they said hello"
    assert extracted[0].episode_timestamp_ms == 1_700_000_000_000
    assert extracted[0].owner_id == "u1"
