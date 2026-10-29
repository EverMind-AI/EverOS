"""extract_foresight strategy — derive Foresights from a fresh MemCell.

Per-sender extraction (mirrors Episode): a foresight is a forward-looking
statement *about* a specific user, so the algo is invoked once per user
sender and each invocation produces foresights whose ``owner_id`` is
already correct. (AtomicFact, by contrast, uses a subject-agnostic
one-call fan-out.)

Per-owner batching: each sender's full foresight list is appended in
one batched ``append_many`` call rather than ``N`` single appends,
dropping IO complexity to ``O(N)`` per owner and narrowing the
per-path lock window.

**Disabled by default** (``enabled=False``). Not because it is broken —
the crash it used to have is fixed below — but because it is one LLM
call per sender per memcell whose output nothing in EverOS reads today:
no search route surfaces foresights, no operator consumes them. Until
something does, running it by default spends tokens on write-only data.
Re-enable per install in ``ome.toml`` (hot-reloaded, ~2s):

.. code-block:: toml

    [strategies.extract_foresight]
    enabled = true

The opt-in has to actually work, so the sender scan below filters on
``isinstance(m, ChatMessage)`` rather than reaching for ``m.role``.
Only ``ChatMessage`` carries ``role``: ``ToolCallRequest`` has
``sender_id`` without it, ``ToolCallResult`` has neither — so the old
attribute test raised ``AttributeError`` on the first tool call, making
the strategy correct on plain user chat and guaranteed to dead-letter on
agent trajectories. everalgo contracts for exactly this mixed input
(``user_memory/_render.chat_messages``: "the caller need not pre-filter;
an AgentMemCell-shaped MemCell is acceptable input"), and every other
user-memory extractor gets it for free by delegating; this strategy was
the one place that hand-rolled the filter. A pure agent trajectory now
yields no senders and returns without an LLM call.

Extraction granularity is a separate, still-open question: this runs per
memcell, while ``atomic_fact`` runs per episode. Moving it needs an
everalgo entry point that does not exist yet — unrelated to the crash,
and not what the default-off above is about.
"""

from __future__ import annotations

from collections import defaultdict

from everalgo.types import ChatMessage
from everalgo.user_memory import ForesightExtractor

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import UserPipelineStarted
from everos.memory.models import Foresight
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def extract_foresight(
    event: UserPipelineStarted, deps: MemoryDeps
) -> list[Event]:
    """Extract foresights per user sender and persist them under each sender."""
    # 1. List the user senders in this memcell.
    memcell = event.memcell
    sender_ids = sorted(
        {
            m.sender_id
            for m in memcell.items
            if isinstance(m, ChatMessage) and m.role == "user"
        }
    )
    extractor = ForesightExtractor(llm=deps.require_llm()) if sender_ids else None

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
    by_owner: dict[str, list[Foresight]] = defaultdict(list)
    for fs in foresights:
        by_owner[fs.owner_id].append(fs)

    # 4. Write each owner's full list with one batched append.
    for owner_id, items in by_owner.items():
        scope = Scope(
            app_id=event.app_id,
            project_id=event.project_id,
            owner_id=owner_id,
            owner_type="user",
        )
        await deps.stores.foresight.append_many(scope, items)

    logger.info(
        "foresights_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(foresights),
        owner_ids=sorted({f.owner_id for f in foresights}),
    )
    return []


EXTRACT_FORESIGHT_SPEC = StrategySpec(
    name="extract_foresight",
    fn=extract_foresight,
    trigger=Immediate(on=[UserPipelineStarted]),
    max_retries=2,
    enabled=False,
)
