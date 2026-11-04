"""Document classification — one LLM call picks a ``CategorySpec.id`` from a closed set.

Pipeline placement: after the doc-level ``summary`` is finalized (post multi-batch
``content_merge``), before ``flatten``. The resulting id is denormalized onto every
emitted ``KnowledgeMemory`` via ``flatten(category_id=...)``.

The taxonomy is caller-owned business data passed in at ingest time. Out-of-set
predictions collapse to ``""`` — the downstream retrieval-side soft boost treats
unclassified hits as a non-bucket and is robust to occasional misses. A reply that
fails the reply model raises ``LLMOutputError``.

NOT exposed in the package ``__all__`` directly — surfaced through ``everos.algo.knowledge``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.knowledge._replies import CategoryReply, ask_llm
from everos.algo.knowledge.prompts.en.category_classify import (
    CATEGORY_CLASSIFY_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.utils import render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import CategorySpec

__all__ = ["aclassify_category"]

logger = logging.getLogger(__name__)


def _render_taxonomy(categories: list[CategorySpec]) -> str:
    """Render the taxonomy block, one entry per line; it is the same for every document of an ingest run."""
    return "\n".join(f"- {c.id}: {c.description}" for c in categories)


async def aclassify_category(
    client: LLMClient,
    title: str,
    doc_summary: str,
    categories: list[CategorySpec],
    *,
    custom_instructions: str | None = None,
) -> str:
    """Classify a document into one ``CategorySpec`` from the supplied taxonomy.

    One LLM call. The taxonomy follows the static rules in the system message, then the
    caller's ``<custom_instructions>`` block when there is one; the title and summary form the
    user message, so the system message stays the same across the documents of one ingest run
    (prompt caching friendly).

    Args:
        client: Caller-injected ``LLMClient``. Algorithm code never holds a client.
        title: Document title — used as classifier input alongside the summary.
        doc_summary: Finalized document summary (post merge in the multi-batch path).
        categories: Closed taxonomy. An empty list short-circuits to ``""`` without
            issuing any LLM call.
        custom_instructions: The caller's instructions, appended after the taxonomy as a
            ``<custom_instructions>`` block, always the system message's last; it adds to the bundled
            rules, never replaces them. Placed verbatim, never rendered as a template. None or a blank
            string adds nothing.

    Returns:
        The chosen ``CategorySpec.id``; ``""`` when categories is empty, the model
        answers ``""``, or the predicted id is not in the closed set.

    Raises:
        LLMOutputError: If the reply is not a valid classification output.
    """
    if not categories:
        return ""

    reply = await ask_llm(
        client,
        system=assemble_system_prompt(
            taxonomy=_render_taxonomy(categories),
            custom_instructions=custom_instructions,
        ),
        user=render_prompt(
            CATEGORY_CLASSIFY_USER_PROMPT,
            None,
            title=title or "Untitled",
            doc_summary=doc_summary,
        ),
        reply_model=CategoryReply,
        operator="knowledge category_classify",
    )
    predicted = reply.category_id.strip()
    if predicted == "":
        return ""

    valid_ids = {c.id for c in categories}
    if predicted not in valid_ids:
        logger.warning(
            "classify_category: predicted id %r not in taxonomy (size=%d); returning empty id",
            predicted,
            len(valid_ids),
        )
        return ""
    return predicted
