"""extract_decision strategy — derive Decisions from a fresh MemCell.

One LLM call per memcell (``DecisionExtractor`` has no ``sender_id``;
every algo ``owner_id`` is ``None``). EverOS then fans the same body
out to every user sender via ``Decision.from_algo``, writes one batched
``append_entries`` per owner, and emits ``DecisionExtracted`` per
written entry so clustering can consume the event bus instead of
racing LanceDB.

An empty list from the extractor is success: no committed trade-off
in the slice, so no md write and no emit.

**Enabled by default** (``enabled=True``). Unlike foresight, Search
will consume decisions; running the extractor on every
``UserPipelineStarted`` is the product path. Disable per install in
``ome.toml`` only for evaluation runs that must not spend the extra
tokens:

.. code-block:: toml

    [strategies.extract_decision]
    enabled = false

The sender scan filters on ``isinstance(m, ChatMessage)`` rather than
reaching for ``m.role``. Only ``ChatMessage`` carries ``role``:
``ToolCallRequest`` has ``sender_id`` without it, ``ToolCallResult``
has neither. A pure agent trajectory yields no user senders and
returns without an LLM call.
"""

from __future__ import annotations

from pathlib import Path

from everalgo.types import ChatMessage
from everalgo.user_memory import DecisionExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import DecisionWriter
from everos.memory.events import DecisionExtracted, UserPipelineStarted
from everos.memory.models import Decision, MemCell
from everos.memory.prompt_slots import PromptLoader

logger = get_logger(__name__)

_writer: DecisionWriter | None = None
_prompt_loader: PromptLoader | None = None


def _config_root() -> Path:
    """Return ``src/everos/config`` (bundled prompt slots)."""
    return Path(__file__).resolve().parents[2] / "config"


def _get_writer() -> DecisionWriter:
    global _writer
    if _writer is None:
        _writer = DecisionWriter(root=MemoryRoot.resolve())
    return _writer


def _get_prompt_loader() -> PromptLoader:
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(_config_root())
    return _prompt_loader


def _unique_user_senders(memcell: MemCell) -> list[str]:
    """Distinct role=user sender_ids, preserving first-seen order.

    Skips non-``ChatMessage`` items (agent trajectories' tool calls
    have no ``role``). Does not sort — order matches Episode
    pipeline ``_unique_user_senders`` so two runs over the same
    memcell fan out identically.
    """
    senders: list[str] = []
    for item in memcell.items:
        if not isinstance(item, ChatMessage) or item.role != "user":
            continue
        sid = item.sender_id
        if sid and sid not in senders:
            senders.append(sid)
    return senders


@offline_strategy(
    name="extract_decision",
    trigger=Immediate(on=[UserPipelineStarted]),
    emits=[DecisionExtracted],
    max_retries=2,
    enabled=True,
)
async def extract_decision(event: UserPipelineStarted, ctx: StrategyContext) -> None:
    owner_ids = _unique_user_senders(event.memcell)
    if not owner_ids:
        logger.info(
            "decisions_extracted",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            count=0,
            owner_ids=[],
        )
        return

    prompt = _get_prompt_loader().load("decision_extract")
    extractor = DecisionExtractor(llm=get_llm_client())
    algo_decisions = await extractor.aextract(event.memcell, prompt=prompt)
    if not algo_decisions:
        logger.info(
            "decisions_extracted",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            count=0,
            owner_ids=owner_ids,
        )
        return

    writer = _get_writer()
    written = 0
    for owner_id in owner_ids:
        decisions = [
            Decision.from_algo(
                algo,
                owner_id=owner_id,
                session_id=event.session_id,
                parent_id=event.memcell_id,
            )
            for algo in algo_decisions
        ]
        items = [_decision_to_entry_body(d) for d in decisions]
        eids = await writer.append_entries(
            owner_id,
            items,
            app_id=event.app_id,
            project_id=event.project_id,
        )
        for d, eid in zip(decisions, eids, strict=True):
            await ctx.emit(
                DecisionExtracted(
                    memcell_id=event.memcell_id,
                    decision_entry_id=eid.format(),
                    title=d.title,
                    decision_text=d.decision,
                    reason=d.reason,
                    impact=d.impact,
                    tags=list(d.tags),
                    decision_timestamp_ms=d.timestamp,
                    owner_id=d.owner_id,
                    session_id=event.session_id,
                    app_id=event.app_id,
                    project_id=event.project_id,
                    source="pipeline",
                )
            )
            written += 1

    logger.info(
        "decisions_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=written,
        owner_ids=owner_ids,
    )


def _decision_to_entry_body(
    d: Decision,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain Decision into ``(inline, sections)`` for md rendering.

    Lives in the strategy (memory) layer rather than the writer (infra
    must not import ``memory``). ``Impact`` is omitted when empty so md
    stays compact; ``tags`` is always written so cascade can parse a list.
    """
    inline: dict[str, object] = {
        "owner_id": d.owner_id,
        "session_id": d.session_id,
        "timestamp": to_iso_format(from_timestamp(d.timestamp)),
        "parent_type": "memcell",
        "parent_id": d.parent_id,
        "tags": list(d.tags),
    }
    sections: dict[str, str] = {
        "Title": d.title,
        "Decision": d.decision,
        "Reason": d.reason,
    }
    if d.impact:
        sections["Impact"] = d.impact
    return inline, sections
