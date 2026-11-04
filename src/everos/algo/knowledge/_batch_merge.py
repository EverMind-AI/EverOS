"""Multi-batch consolidation of long-document extraction results.

When ``KnowledgeExtractor`` processes a document that exceeds a single
LLM context window, the topic-tree prompt runs once per batch and emits
independent partial results. This module wires those partials into a
single coherent extraction:

* ``amerge_doc_summary`` — calls the ``content_merge`` prompt to consolidate
  per-batch ``summary`` fields into one document-level summary. The
  prompt also returns subject / keywords; ``KnowledgeMemory`` has no
  field for them so they are dropped here. ``content_labels`` are not
  produced by this prompt — the caller union-dedupes them in code.

* ``amerge_topics`` — calls the ``topic_merge`` prompt to identify true
  duplicate topics across batch boundaries (e.g. the same event surfaced
  in two batches because it straddled the split). Hierarchy is flattened
  with parent-topic context so the LLM only reasons about siblings; the
  LLM reports duplicate groups by index, and the tree is rebuilt in
  deterministic code so the LLM never has to manage tree shape.

Short-circuits (mirrors ``memsys_enterprise``):

* ``amerge_doc_summary`` returns the only batch's summary verbatim when
  there is just one batch.
* ``amerge_topics`` returns the input forest unchanged when fewer than
  ``MIN_TOPICS_FOR_MERGE`` flattened topics exist — too small to benefit
  from de-duplication.

Failure behavior: a reply that fails its reply model raises
``LLMOutputError``; the client's own exceptions propagate unchanged. A valid
reply whose merged summary is blank falls back to the first batch's summary,
and merge groups naming fewer than two in-range indices are ignored.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from everos.algo.knowledge._replies import ContentMergeReply, TopicMergeReply, ask_llm
from everos.algo.knowledge.prompts.en.content_merge import (
    CONTENT_MERGE_SYSTEM_PROMPT,
    CONTENT_MERGE_USER_PROMPT,
)
from everos.algo.knowledge.prompts.en.topic_merge import (
    TOPIC_MERGE_SYSTEM_PROMPT,
    TOPIC_MERGE_USER_PROMPT,
)
from everos.algo.utils import append_custom_instructions, render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

__all__ = [
    "MIN_TOPICS_FOR_MERGE",
    "amerge_doc_summary",
    "amerge_topics",
]

logger = logging.getLogger(__name__)

# Below this many post-flatten topics, semantic de-duplication is not worth an LLM call.
MIN_TOPICS_FOR_MERGE = 4


async def amerge_doc_summary(
    client: LLMClient,
    title: str,
    batch_results: list[dict[str, Any]],
    *,
    custom_instructions: str | None = None,
) -> str:
    """Consolidate per-batch document-level summaries via one LLM call.

    Args:
        client: The client the call goes through.
        title: Document title.
        batch_results: Per-batch topic-tree replies as dicts; ``summary`` / ``subject`` / ``keywords`` are read.
        custom_instructions: The caller's instructions, appended to the rules as a ``<custom_instructions>`` block,
            always the system message's last. Placed verbatim, never rendered as a template; None or a blank string
            adds nothing.

    Returns:
        The merged summary; the first batch's summary when there is one batch or the merged one is blank.

    Raises:
        LLMOutputError: If the reply is not a valid content-merge output.
    """
    if not batch_results:
        return ""
    fallback: str = batch_results[0].get("summary", "")
    if len(batch_results) == 1:
        return fallback

    lines: list[str] = []
    for i, r in enumerate(batch_results):
        lines.append(
            f"{i + 1}. Summary: {r.get('summary', '')}\n"
            f"   Subject: {r.get('subject', '')}\n"
            f"   Keywords: {r.get('keywords', [])}",
        )
    reply = await ask_llm(
        client,
        system=append_custom_instructions(
            CONTENT_MERGE_SYSTEM_PROMPT, custom_instructions
        ),
        user=render_prompt(
            CONTENT_MERGE_USER_PROMPT,
            None,
            title=title,
            count=len(lines),
            partial_results="\n\n".join(lines),
        ),
        reply_model=ContentMergeReply,
        operator="knowledge content_merge",
    )
    if not reply.summary.strip():
        logger.warning(
            "multi-batch content-merge: empty summary field; falling back to batch-1 summary"
        )
        return fallback
    return reply.summary


async def amerge_topics(
    client: LLMClient,
    title: str,
    all_topics: list[dict[str, Any]],
    *,
    custom_instructions: str | None = None,
) -> list[dict[str, Any]]:
    """De-duplicate topics across batch boundaries via one LLM call.

    Args:
        client: The client the call goes through.
        title: Document title.
        all_topics: The concatenated per-batch topic forests.
        custom_instructions: The caller's instructions, appended to the rules as a ``<custom_instructions>`` block,
            always the system message's last. Placed verbatim, never rendered as a template; None or a blank string
            adds nothing.

    Returns:
        The rebuilt forest; ``all_topics`` itself when it is too small to merge or no group applies.

    Raises:
        LLMOutputError: If the reply is not a valid topic-merge output.
    """
    if not all_topics:
        return all_topics

    flat_list, parent_indices = _flatten_for_merge(all_topics)
    if len(flat_list) < MIN_TOPICS_FOR_MERGE:
        return all_topics

    indexed = [
        {
            "index": entry["index"],
            "topic": entry["topic"],
            "summary": entry["summary"],
            "parent_topic": entry.get("parent_topic"),
        }
        for entry in flat_list
    ]
    reply = await ask_llm(
        client,
        system=append_custom_instructions(
            TOPIC_MERGE_SYSTEM_PROMPT, custom_instructions
        ),
        user=render_prompt(
            TOPIC_MERGE_USER_PROMPT,
            None,
            title=title,
            all_topics_json=json.dumps(indexed, ensure_ascii=False, indent=2),
        ),
        reply_model=TopicMergeReply,
        operator="knowledge topic_merge",
    )
    if not reply.merges:
        return all_topics

    merged_flat, index_remap = _apply_merge_groups_with_remap(
        flat_list, [group.model_dump() for group in reply.merges]
    )
    if not index_remap:
        return all_topics

    return _rebuild_hierarchy(merged_flat, parent_indices, index_remap)


# ── pure-compute helpers ─────────────────────────────────────────────


def _flatten_for_merge(
    nested_topics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int | None]]:
    """DFS-flatten the topic forest, recording parent indices.

    Each entry carries ``index``, ``topic``, ``summary``, ``parent_topic``
    (the parent's topic name; ``None`` for forest roots), plus the
    original ``block_refs`` / ``content_labels`` so callers can
    reconstitute the surviving node when a merge applies.
    """
    flat: list[dict[str, Any]] = []
    parents: dict[int, int | None] = {}

    def _visit(
        topic: dict[str, Any], parent_idx: int | None, parent_name: str | None
    ) -> None:
        my_idx = len(flat)
        parents[my_idx] = parent_idx
        flat.append(
            {
                "index": my_idx,
                "topic": topic.get("topic", ""),
                "summary": topic.get("summary", ""),
                "parent_topic": parent_name,
                "block_refs": topic.get("block_refs", ""),
                "content_labels": list(topic.get("content_labels", []) or []),
            },
        )
        for child in topic.get("children", []) or []:
            _visit(child, my_idx, topic.get("topic", ""))

    for root in nested_topics:
        _visit(root, None, None)
    return flat, parents


def _apply_merge_groups_with_remap(
    flat_topics: list[dict[str, Any]],
    merges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Apply LLM-reported merge groups, returning ``(survivors, remap)``.

    ``remap`` maps every absorbed index → its surviving index. The caller
    uses this when rebuilding the tree to redirect children whose
    original parent has been absorbed.
    """
    merged_indices: set[int] = set()
    index_remap: dict[int, int] = {}
    merge_results: dict[int, dict[str, Any]] = {}

    for group in merges:
        raw_indices = group.get("indices", []) or []
        indices = [
            i for i in raw_indices if isinstance(i, int) and 0 <= i < len(flat_topics)
        ]
        if len(indices) < 2:
            continue

        surviving = min(indices)
        joined_refs: list[str] = []
        merged_labels: list[str] = []
        for idx in indices:
            refs = flat_topics[idx].get("block_refs", "")
            if refs:
                joined_refs.append(str(refs))
            merged_labels.extend(flat_topics[idx].get("content_labels", []) or [])
            merged_indices.add(idx)
            index_remap[idx] = surviving

        merge_results[surviving] = {
            "index": surviving,
            "topic": group.get("topic") or flat_topics[indices[0]].get("topic", ""),
            "summary": group.get("summary")
            or flat_topics[indices[0]].get("summary", ""),
            "block_refs": ",".join(joined_refs),
            "content_labels": list(dict.fromkeys(merged_labels)),
            "parent_topic": flat_topics[surviving].get("parent_topic"),
        }

    return _build_survivors(flat_topics, merged_indices, merge_results), index_remap


def _build_survivors(
    flat_topics: list[dict[str, Any]],
    merged_indices: set[int],
    merge_results: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the surviving topic list after merges, preferring merge results over originals."""
    survivors: list[dict[str, Any]] = []
    for i, topic in enumerate(flat_topics):
        if i in merge_results:
            survivors.append(merge_results[i])
        elif i not in merged_indices:
            survivors.append(topic)
    return survivors


def _rebuild_hierarchy(
    merged_flat: list[dict[str, Any]],
    parent_indices: dict[int, int | None],
    index_remap: dict[int, int],
) -> list[dict[str, Any]]:
    """Rebuild a nested topic forest from the flat list + parent-pointer table.

    Children orphaned by an absorbed parent are reparented to the
    surviving topic via ``index_remap``. Forest roots are topics whose
    resolved parent is ``None`` or no longer present.
    """
    by_index: dict[int, dict[str, Any]] = {}
    for entry in merged_flat:
        entry.setdefault("children", [])
        by_index[entry["index"]] = entry

    roots: list[dict[str, Any]] = []
    for entry in merged_flat:
        raw_parent = parent_indices.get(entry["index"])
        if raw_parent is None:
            real_parent: int | None = None
        else:
            real_parent = index_remap.get(raw_parent, raw_parent)
        # Self-loop guard: an absorbed ancestor being remapped to entry itself
        # would create a cycle. Treat as a root in that pathological case.
        if (
            real_parent is None
            or real_parent == entry["index"]
            or real_parent not in by_index
        ):
            roots.append(entry)
        else:
            by_index[real_parent]["children"].append(entry)

    def _clean(topics: list[dict[str, Any]]) -> None:
        for t in topics:
            t.pop("index", None)
            t.pop("parent_topic", None)
            _clean(t.get("children", []))

    _clean(roots)
    return roots
