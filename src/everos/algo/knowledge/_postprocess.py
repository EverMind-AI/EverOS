"""LLM-driven post-processing of the extracted topic tree.

Stage 4 of the KnowledgeExtractor pipeline — *two* conditional LLM passes:

1. ``asplit_unsplit_leaves`` (uses the ``topic_split`` prompt): leaves with
   multiple sub-headings that the initial extraction left intact get
   inspected by the LLM, which may decide to re-split them into a
   parent / children hierarchy.

2. ``aassign_uncovered_blocks`` (uses the ``uncovered_assign`` prompt):
   non-separator, non-empty atoms that no topic claimed are attached to
   the most relevant existing topic.

Each pass is a no-op when its detector returns empty (short / fully-
covered documents bypass the LLM entirely). Detectors are pure compute;
appliers are async. A reply that fails its reply model raises ``LLMOutputError``;
decisions that name an unknown topic or an out-of-range index are skipped.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from everos.algo.knowledge._replies import (
    TopicSplitReply,
    UncoveredAssignReply,
    ask_llm,
)
from everos.algo.knowledge._topic_build import (
    collect_all_block_refs,
    parse_segment_ranges,
)
from everos.algo.knowledge.prompts.en.topic_split import (
    TOPIC_SPLIT_SYSTEM_PROMPT,
    TOPIC_SPLIT_USER_PROMPT,
)
from everos.algo.knowledge.prompts.en.uncovered_assign import (
    UNCOVERED_ASSIGN_SYSTEM_PROMPT,
    UNCOVERED_ASSIGN_USER_PROMPT,
)
from everos.algo.utils import append_custom_instructions, render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

__all__ = [
    "POSTPROCESS_MIN_AVG_CHARS",
    "POSTPROCESS_MIN_TOTAL_CHARS",
    "aassign_uncovered_blocks",
    "apostprocess_topics",
    "asplit_unsplit_leaves",
    "detect_uncovered_blocks",
    "detect_unsplit_leaves",
]

# Minimum char thresholds for triggering a split pass on a leaf.
POSTPROCESS_MIN_TOTAL_CHARS = 2000
POSTPROCESS_MIN_AVG_CHARS = 500

_HEADING_RE = re.compile(r"^#{2,5}\s+")
_SEPARATOR_RE = re.compile(r"^[-*=]{3,}$")


# ── pure-compute detectors ───────────────────────────────────────────


def _scan_unsplit_leaves(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
    path: list[int],
    results: list[dict[str, Any]],
) -> None:
    """DFS for leaf topics that need splitting based on heading count + size."""
    for i, t in enumerate(topics):
        current_path = [*path, i]
        children = t.get("children", [])
        if children:
            _scan_unsplit_leaves(children, atom_map, current_path, results)
            continue

        refs = parse_segment_ranges(t.get("block_refs", ""))
        if not refs:
            continue

        heading_ids = [
            r for r in refs if r in atom_map and _HEADING_RE.match(atom_map[r].strip())
        ]
        if len(heading_ids) < 2:
            continue

        total_chars = sum(len(atom_map.get(r, "")) for r in refs)
        avg_chars = total_chars / len(heading_ids)
        if (
            total_chars >= POSTPROCESS_MIN_TOTAL_CHARS
            and avg_chars >= POSTPROCESS_MIN_AVG_CHARS
        ):
            results.append(
                {
                    "topic_name": t.get("topic", ""),
                    "path": current_path,
                    "block_ids": refs,
                    "heading_ids": heading_ids,
                    "total_chars": total_chars,
                },
            )


def detect_unsplit_leaves(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> list[dict[str, Any]]:
    """Find leaf topics with >= 2 sub-headings and enough content to warrant splitting."""
    results: list[dict[str, Any]] = []
    _scan_unsplit_leaves(topics, atom_map, [], results)
    return results


def detect_uncovered_blocks(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> list[dict[str, Any]]:
    """Find non-separator, non-empty atoms that no topic claimed."""
    covered = collect_all_block_refs(topics)
    uncovered: list[dict[str, Any]] = []
    for bid, text in sorted(atom_map.items()):
        if bid in covered:
            continue
        stripped = text.strip()
        if not stripped or _SEPARATOR_RE.match(stripped):
            continue
        uncovered.append({"block_id": bid, "text": text})
    return uncovered


# ── tree navigation helpers ──────────────────────────────────────────


def _replace_topic_at_path(
    topics: list[dict[str, Any]],
    path: list[int],
    replacement: dict[str, Any],
) -> None:
    """Replace the topic at ``path`` with a split result, preserving identity fields.

    The original topic name, summary, and content_labels are kept; only
    block_refs and children come from the replacement.
    """
    container = topics
    for idx in path[:-1]:
        if idx >= len(container):
            return
        container = container[idx].get("children", [])
    final_idx = path[-1]
    if final_idx >= len(container):
        return
    original = container[final_idx]
    container[final_idx] = {
        "topic": original.get("topic", ""),
        "summary": original.get("summary", ""),
        "block_refs": replacement.get("block_refs", ""),
        "content_labels": original.get("content_labels", []),
        "children": replacement.get("children", []),
    }


def _get_topic_at_path(
    topics: list[dict[str, Any]],
    path: list[int],
) -> dict[str, Any] | None:
    """Navigate the nested tree by index path; returns ``None`` if out of range."""
    container = topics
    for idx in path[:-1]:
        if idx >= len(container):
            return None
        container = container[idx].get("children", [])
    final_idx = path[-1]
    return container[final_idx] if final_idx < len(container) else None


def _flatten_topics_for_assign(
    topics: list[dict[str, Any]],
    path: list[int],
    result: list[dict[str, Any]],
) -> None:
    """Build a flat list of topics with display strings for the assign-prompt."""
    for i, t in enumerate(topics):
        current_path = [*path, i]
        topic_name = t.get("topic", "")
        refs = parse_segment_ranges(t.get("block_refs", ""))
        children = t.get("children", [])

        if refs or not children:
            display = topic_name
            if refs:
                display += f" (blocks {min(refs)}-{max(refs)})"
            result.append({"display": display, "path": current_path})

        if children:
            _flatten_topics_for_assign(children, current_path, result)


# ── LLM passes ───────────────────────────────────────────────────────


def _render_split_sections(
    unsplit: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> str:
    sections: list[str] = []
    for u in unsplit:
        blocks_text = "\n".join(
            f"{bid}: {atom_map[bid][:200]}" for bid in u["block_ids"] if bid in atom_map
        )
        headings_text = "\n".join(
            f"  [{hid}] {atom_map[hid][:80]}"
            for hid in u["heading_ids"]
            if hid in atom_map
        )
        sections.append(
            f"--- Topic: {u['topic_name']} ({len(u['block_ids'])} blocks, "
            f"{u['total_chars']} chars) ---\n"
            f"Blocks:\n{blocks_text}\n\n"
            f"Sub-headings found:\n{headings_text}",
        )
    return "\n\n".join(sections)


async def asplit_unsplit_leaves(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    unsplit: list[dict[str, Any]],
    atom_map: dict[int, str],
    *,
    custom_instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Ask the LLM whether to split each leaf and apply the decisions in place.

    Args:
        llm: The client the call goes through.
        title: Document title.
        topics_data: The topic forest, mutated in place.
        unsplit: The leaves :func:`detect_unsplit_leaves` found.
        atom_map: Atom text by block id.
        custom_instructions: The caller's instructions, appended to the rules as a ``<custom_instructions>`` block,
            always the system message's last. Placed verbatim, never rendered as a template; None or a blank string
            adds nothing.

    Returns:
        ``topics_data``, each leaf the model split replaced by its parent / children shape.

    Raises:
        LLMOutputError: If the reply is not a valid topic-split output.
    """
    user = render_prompt(
        TOPIC_SPLIT_USER_PROMPT,
        None,
        title=title,
        count=len(unsplit),
        topics_section=_render_split_sections(unsplit, atom_map),
    )
    reply = await ask_llm(
        llm,
        system=append_custom_instructions(
            TOPIC_SPLIT_SYSTEM_PROMPT, custom_instructions
        ),
        user=user,
        reply_model=TopicSplitReply,
        operator="knowledge topic_split",
    )
    split_map = {
        r.original_topic: r.model_dump()
        for r in reply.results
        if r.should_split and r.children
    }

    for u in unsplit:
        replacement = split_map.get(u["topic_name"])
        if replacement is not None:
            _replace_topic_at_path(topics_data, u["path"], replacement)

    return topics_data


async def aassign_uncovered_blocks(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    uncovered: list[dict[str, Any]],
    atom_map: dict[int, str],
    *,
    custom_instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Ask the LLM to attach orphan atoms to existing topics; append to ``block_refs``.

    Args:
        llm: The client the call goes through.
        title: Document title.
        topics_data: The topic forest, mutated in place.
        uncovered: The atoms :func:`detect_uncovered_blocks` found.
        atom_map: Atom text by block id.
        custom_instructions: The caller's instructions, appended to the rules as a ``<custom_instructions>`` block,
            always the system message's last. Placed verbatim, never rendered as a template; None or a blank string
            adds nothing.

    Returns:
        ``topics_data`` with each assigned block appended to its topic's ``block_refs``.

    Raises:
        LLMOutputError: If the reply is not a valid uncovered-assign output.
    """
    flat_topics: list[dict[str, Any]] = []
    _flatten_topics_for_assign(topics_data, [], flat_topics)

    user = render_prompt(
        UNCOVERED_ASSIGN_USER_PROMPT,
        None,
        title=title,
        topic_count=len(flat_topics),
        topics_text="\n".join(
            f"{i}: {t['display']}" for i, t in enumerate(flat_topics)
        ),
        block_count=len(uncovered),
        uncovered_text="\n".join(
            f"block {u['block_id']}: {u['text'][:300]}" for u in uncovered
        ),
    )
    reply = await ask_llm(
        llm,
        system=append_custom_instructions(
            UNCOVERED_ASSIGN_SYSTEM_PROMPT, custom_instructions
        ),
        user=user,
        reply_model=UncoveredAssignReply,
        operator="knowledge uncovered_assign",
    )
    for assignment in reply.assignments:
        if not 0 <= assignment.topic_index < len(flat_topics):
            continue

        target = _get_topic_at_path(
            topics_data, flat_topics[assignment.topic_index]["path"]
        )
        if target is None:
            continue

        existing = target.get("block_refs", "")
        target["block_refs"] = (
            f"{existing},{assignment.block_id}"
            if existing
            else str(assignment.block_id)
        )

    return topics_data


async def apostprocess_topics(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    atoms: list[tuple[int, str]],
    *,
    custom_instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Run both LLM passes if their detectors find anything to fix.

    Returns ``topics_data`` (possibly mutated in place). Each pass is
    skipped when its detector returns empty; ``custom_instructions`` closes
    the system message of each pass that runs.

    Raises:
        LLMOutputError: If a pass's reply fails its reply model.
    """
    atom_map = dict(atoms)

    unsplit = detect_unsplit_leaves(topics_data, atom_map)
    if unsplit:
        topics_data = await asplit_unsplit_leaves(
            llm,
            title,
            topics_data,
            unsplit,
            atom_map,
            custom_instructions=custom_instructions,
        )

    uncovered = detect_uncovered_blocks(topics_data, atom_map)
    if uncovered:
        topics_data = await aassign_uncovered_blocks(
            llm,
            title,
            topics_data,
            uncovered,
            atom_map,
            custom_instructions=custom_instructions,
        )

    return topics_data
