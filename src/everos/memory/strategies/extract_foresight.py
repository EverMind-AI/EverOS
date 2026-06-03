"""extract_foresight strategy — derive Foresights from a fresh MemCell.

Per-sender extraction (mirrors Episode): a foresight is a forward-looking
statement *about* a specific user, so the algo is invoked once per user
sender and each invocation produces foresights whose ``owner_id`` is
already correct. (AtomicFact, by contrast, uses a subject-agnostic
one-call fan-out.)

Per-owner batching: each sender's full foresight list is appended in
one batched ``append_entries`` call rather than ``N`` single appends,
dropping IO complexity to ``O(N)`` per owner and narrowing the
per-path lock window.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from everalgo.user_memory import ForesightExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import ForesightWriter
from everos.memory.events import UserPipelineStarted
from everos.memory.models import Foresight

logger = get_logger(__name__)

_writer: ForesightWriter | None = None


def _get_writer() -> ForesightWriter:
    global _writer
    if _writer is None:
        _writer = ForesightWriter(root=MemoryRoot.default())
    return _writer


@offline_strategy(
    name="extract_foresight",
    trigger=Immediate(on=[UserPipelineStarted]),
    emits=[],
    max_retries=2,
)
async def extract_foresight(event: UserPipelineStarted, ctx: StrategyContext) -> None:
    # 1. List the user senders in this memcell.
    memcell = event.memcell
    sender_ids = sorted({m.sender_id for m in memcell.items if m.role == "user"})
    extractor = ForesightExtractor(llm=get_llm_client()) if sender_ids else None

    # 2. Run the LLM extractor once per sender (prompt is per-sender).
    foresights: list[Foresight] = []
    for sid in sender_ids:
        algo_foresights = await extractor.aextract(memcell, sender_id=sid)
        foresights.extend(
            Foresight.from_algo(
                algo_fs,
                session_id=event.session_id,
                parent_id=event.memcell_id,
            )
            for algo_fs in algo_foresights
        )

    # 3. Group foresights by owner so each sender's full list lands in one
    #    batched write.
    by_owner: dict[str, list[tuple[Mapping[str, object], Mapping[str, str]]]] = (
        defaultdict(list)
    )
    for fs in foresights:
        by_owner[fs.owner_id].append(_foresight_to_entry_body(fs))

    # 4. Write each owner's full list with one batched append_entries.
    writer = _get_writer()
    for owner_id, items in by_owner.items():
        await writer.append_entries(
            owner_id, items, app_id=event.app_id, project_id=event.project_id
        )

    logger.info(
        "foresights_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(foresights),
        owner_ids=sorted({f.owner_id for f in foresights}),
    )


def _foresight_to_entry_body(
    fs: Foresight,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain Foresight into ``(inline, sections)`` for md rendering.

    Mirrors ``_episode_to_entry_body`` / ``_atomic_fact_to_entry_body``.
    Optional time-window fields (``start_time`` / ``end_time`` /
    ``duration_days``) are emitted only when set so md stays compact.
    """
    inline: dict[str, object] = {
        "owner_id": fs.owner_id,
        "session_id": fs.session_id,
        "timestamp": to_iso_format(from_timestamp(fs.timestamp)),
        "parent_type": "memcell",
        "parent_id": fs.parent_id,
    }
    if fs.start_time:
        inline["start_time"] = fs.start_time
    if fs.end_time:
        inline["end_time"] = fs.end_time
    if fs.duration_days is not None:
        inline["duration_days"] = fs.duration_days
    sections = {"Foresight": fs.foresight, "Evidence": fs.evidence}
    return inline, sections
