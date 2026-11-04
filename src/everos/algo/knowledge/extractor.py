"""KnowledgeExtractor — ParsedContent → list[KnowledgeMemory].

Orchestrates the four-stage knowledge extraction pipeline:

1. **Preprocess + atomize** (``_block_split``): normalize parsed text into
   numbered atom blocks (paragraphs, merged tables, merged lists).
2. **Topic-tree LLM extraction** (``prompts.en.topic_tree_extract``): one
   LLM call per token-bounded batch produces a nested ``topics`` JSON
   plus document-level ``summary`` / ``content_labels``.
3. **Cross-batch merge** (``_batch_merge``, multi-batch only): one LLM
   call via ``prompts.en.content_merge`` consolidates per-batch summaries,
   then one LLM call via ``prompts.en.topic_merge`` de-duplicates topics
   that straddle a batch boundary. Skipped entirely for single-batch
   documents.
4. **Post-processing** (``_postprocess``): two conditional LLM passes —
   split leaves that still contain multiple sub-headings, and assign any
   blocks no topic claimed.
5. **Tree assembly** (``_topic_build`` → ``_doc_root`` → ``_flatten``):
   materialize a ``TopicClip`` forest, simplify it (collapse trivial
   children + deduplicate parent refs), wrap under a synthetic doc root
   carrying the title + doc-level summary, and DFS-flatten into the
   public ``list[KnowledgeMemory]`` output.

Every LLM call is held to a strict reply model (``_replies``); a reply that does not validate raises
``LLMOutputError`` and stops the extraction, so no partial tree is ever returned in its place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from everos.algo.knowledge._batch_merge import amerge_doc_summary, amerge_topics
from everos.algo.knowledge._block_split import (
    format_numbered_paragraphs,
    preprocess_content,
    split_and_batch_content,
    split_content_to_blocks,
)
from everos.algo.knowledge._classify import aclassify_category
from everos.algo.knowledge._doc_root import build_doc_root
from everos.algo.knowledge._flatten import flatten
from everos.algo.knowledge._postprocess import apostprocess_topics
from everos.algo.knowledge._replies import TopicTreeReply, ask_llm
from everos.algo.knowledge._topic_build import build_topic_clips
from everos.algo.knowledge.prompts.en.topic_tree_extract import (
    TOPIC_TREE_EXTRACT_SYSTEM_PROMPT,
    TOPIC_TREE_EXTRACT_USER_PROMPT,
)
from everos.algo.utils import append_custom_instructions, render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import CategorySpec, KnowledgeMemory, ParsedContent

__all__ = ["KnowledgeExtractor"]

logger = logging.getLogger(__name__)


async def _aextract_batch(
    client: LLMClient,
    title: str,
    batch: list[tuple[int, str]],
    custom_instructions: str | None,
) -> dict[str, Any]:
    """Single LLM call for one non-empty atom batch; returns the validated reply as a plain dict.

    Raises:
        LLMOutputError: If the reply is not a valid topic-tree output.
    """
    user = render_prompt(
        TOPIC_TREE_EXTRACT_USER_PROMPT,
        None,
        title=title,
        count=len(batch),
        max_id=batch[-1][0],
        numbered_paragraphs=format_numbered_paragraphs(batch),
    )
    reply = await ask_llm(
        client,
        system=append_custom_instructions(
            TOPIC_TREE_EXTRACT_SYSTEM_PROMPT, custom_instructions
        ),
        user=user,
        reply_model=TopicTreeReply,
        operator="knowledge topic_tree_extract",
    )
    return reply.model_dump()


async def _arun_extraction_batches(
    client: LLMClient,
    text: str,
    title: str,
    custom_instructions: str | None,
) -> tuple[list[tuple[int, str]], list[dict[str, Any]]] | None:
    """Preprocess, atomize, batch, and run per-batch LLM extraction.

    Returns ``(atoms, batch_results)``, or ``None`` when the input yields no atoms.

    Raises:
        LLMOutputError: If a batch's reply is not a valid topic-tree output.
    """
    text = preprocess_content(text)
    atoms = split_content_to_blocks(text)
    if not atoms:
        logger.warning("knowledge.aextract: empty input, no atoms")
        return None

    batch_results = [
        await _aextract_batch(client, title, batch, custom_instructions)
        for batch in split_and_batch_content(atoms)
    ]
    return atoms, batch_results


def _consolidate_batch_results(
    batch_results: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Combine per-batch JSON dicts into ``(doc_summary, doc_labels, all_topics)``.

    Pure-compute concatenation only. The single-batch path uses fields
    verbatim. The multi-batch path concatenates topics and union-dedupes
    labels here, while the caller drives the LLM passes in
    ``_batch_merge`` to produce the final document summary and to fold
    duplicate topics across batch boundaries.
    """
    if len(batch_results) == 1:
        only = batch_results[0]
        return (
            only.get("summary", ""),
            list(only.get("content_labels", []) or []),
            list(only.get("topics", []) or []),
        )

    doc_summary = batch_results[0].get("summary", "")
    seen_labels: set[str] = set()
    doc_labels: list[str] = []
    all_topics: list[dict[str, Any]] = []
    for r in batch_results:
        for label in r.get("content_labels", []) or []:
            if label not in seen_labels:
                seen_labels.add(label)
                doc_labels.append(label)
        all_topics.extend(r.get("topics", []) or [])
    return doc_summary, doc_labels, all_topics


class KnowledgeExtractor:
    """Extracts a flattened topic tree from one document.

    Construct with the LLM client (``KnowledgeExtractor(llm=client)``); the
    same instance can extract many documents. Add to the bundled rules per call
    via ``custom_instructions=``.

    Args:
        llm: Caller-injected ``LLMClient``.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        doc_id: str = "",
        title: str = "",
        categories: list[CategorySpec] | None = None,
        category_id: str = "",
        custom_instructions: str | None = None,
    ) -> list[KnowledgeMemory]:
        """Extract a flattened topic tree from ``parsed``.

        Args:
            parsed: Single-document parser output; ``parsed.text`` carries the
                normalized content to segment.
            doc_id: Identifier copied into every emitted ``KnowledgeMemory.doc_id``.
                ``ParsedContent`` no longer carries an id, so the caller supplies
                it here.
            title: Document title used as the synthetic doc-root topic; falls back
                to ``"Untitled"`` when empty.
            categories: Optional classification taxonomy. When non-empty and
                ``category_id`` is not pre-supplied, the extractor runs a single
                LLM classification call against the finalized doc summary and
                denormalizes the chosen id onto every emitted node. The taxonomy
                is caller-owned business data; the extractor ships no default.
            category_id: Pre-known category id. Non-empty values short-circuit
                the classifier and are used verbatim. ``""`` (default) means
                "ask the classifier if ``categories`` is also provided".
            custom_instructions: The caller's instructions, appended to every LLM call of the
                extraction (each topic-tree batch, the content and topic merges, classification,
                leaf split, uncovered-block assignment) as a ``<custom_instructions>`` block, always
                the system message's last; it adds to the bundled rules, never replaces them. Placed
                verbatim, never rendered as a template. None or a blank string adds nothing.

        Returns:
            ``list[KnowledgeMemory]`` — index 0 is the synthetic doc-root
            (``depth=0``, ``parent_index=None``); subsequent entries are
            DFS-ordered descendants. Returns an empty list when the input
            yields no atom blocks.

        Raises:
            LLMOutputError: If any LLM reply along the pipeline (topic tree, merges,
                classification, post-processing) fails its reply model.
        """
        client = self._llm
        title = title or "Untitled"

        result = await _arun_extraction_batches(
            client, parsed.text, title, custom_instructions
        )
        if result is None:
            return []
        atoms, batch_results = result

        doc_summary, doc_labels, all_topics = _consolidate_batch_results(batch_results)

        if len(batch_results) > 1:
            doc_summary = await amerge_doc_summary(
                client, title, batch_results, custom_instructions=custom_instructions
            )
            all_topics = await amerge_topics(
                client, title, all_topics, custom_instructions=custom_instructions
            )

        if not category_id and categories:
            category_id = await aclassify_category(
                client,
                title,
                doc_summary,
                categories,
                custom_instructions=custom_instructions,
            )

        all_topics = await apostprocess_topics(
            client, title, all_topics, atoms, custom_instructions=custom_instructions
        )
        forest = build_topic_clips(all_topics, atoms)
        root = build_doc_root(
            forest,
            doc_title=title,
            doc_summary=doc_summary,
            doc_content_labels=doc_labels,
        )
        return flatten(root, doc_id=doc_id, category_id=category_id)
