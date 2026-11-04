"""TopicClip tree construction from LLM JSON output.

Pure-compute stage 2 of the extraction pipeline. No LLM calls, no I/O.
Parses the JSON ``topics`` array emitted by the topic-tree prompt,
materialises ``TopicClip`` dataclasses with hierarchy and atom-joined
content, then applies two simplifications:

* ``collapse_trivial_children`` — merge over-split or heading-only
  children back into the parent.
* ``deduplicate_parent_refs`` — remove ``block_refs`` from a parent that
  already appear in any of its descendants.

The resulting forest is the input for both ``_postprocess`` (LLM-driven
leaf-splitting and orphan assignment) and ``_flatten`` (DFS to the public
``list[KnowledgeMemory]`` output).

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TOPIC_MAX_DEPTH",
    "TopicClip",
    "build_topic_clips",
    "collapse_trivial_children",
    "collect_all_block_refs",
    "deduplicate_parent_refs",
    "parse_segment_ranges",
]

logger = logging.getLogger(__name__)

# Max recursion depth for nested topics. The LLM occasionally produces runaway
# nesting; this acts as a structural safety cap.
TOPIC_MAX_DEPTH = 5


@dataclass
class TopicClip:
    """Internal intermediate tree node used between extraction and flattening.

    Mirrors the shape produced by the topic-tree prompt. Not a public type:
    the public output of ``KnowledgeExtractor`` is ``list[KnowledgeMemory]``,
    produced by ``_flatten``.

    Attributes:
        topic: Section / topic name extracted by the LLM.
        summary: One-sentence summary of this section.
        content: Original prose joined from atoms via ``block_refs``.
        block_refs: Atom indices composing this node's content.
        content_labels: Free-form labels (safety / privacy / topical).
        children: Nested sub-topics in the hierarchy.
    """

    topic: str
    summary: str
    content: str = ""
    block_refs: list[int] = field(default_factory=list)
    content_labels: list[str] = field(default_factory=list)
    children: list[TopicClip] = field(default_factory=list)


# ── parsing LLM block-ref strings ────────────────────────────────────


def parse_segment_ranges(ranges_str: str) -> list[int]:
    """Parse compressed range notation, e.g. ``'1-3,5,8-10' -> [1, 2, 3, 5, 8, 9, 10]``.

    Tolerant of malformed parts: each unparseable segment is logged at WARNING
    and skipped, so a partial parse always succeeds rather than raising. Inner
    whitespace inside a range (``'3 - 5'``) is accepted since ``int`` strips
    whitespace. Returns an empty list for empty / falsy input.
    """
    if not ranges_str:
        return []

    out: list[int] = []
    for raw in str(ranges_str).split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                out.extend(range(int(start), int(end) + 1))
            except (ValueError, TypeError):
                logger.warning("failed to parse range part: %r", part)
        else:
            try:
                out.append(int(part))
            except (ValueError, TypeError):
                logger.warning("failed to parse range id: %r", part)
    return out


# ── tree construction ────────────────────────────────────────────────


def _rebuild_content(refs: list[int], atom_map: dict[int, str]) -> str:
    return "\n".join(atom_map[r] for r in refs if r in atom_map)


def _build_one_topic_clip(
    t: dict[str, Any],
    atom_map: dict[int, str],
    depth: int,
) -> TopicClip:
    """Build a single TopicClip from a raw LLM dict, recursing into children."""
    refs = parse_segment_ranges(t.get("block_refs", ""))
    content = _rebuild_content(refs, atom_map)
    children_data = t.get("children", []) if depth < TOPIC_MAX_DEPTH else []
    children = [_build_one_topic_clip(c, atom_map, depth + 1) for c in children_data]
    return TopicClip(
        topic=t.get("topic", ""),
        summary=t.get("summary", ""),
        content=content,
        block_refs=refs,
        content_labels=t.get("content_labels", []),
        children=children,
    )


def build_topic_clips(
    topics_data: list[dict[str, Any]],
    atoms: list[tuple[int, str]],
) -> list[TopicClip]:
    """Convert the LLM ``topics`` JSON list into a TopicClip forest.

    Joins atom text via the ``block_refs`` of each clip and recurses into
    nested ``children``. Applies ``collapse_trivial_children`` and
    ``deduplicate_parent_refs`` as a final simplification pass.
    """
    atom_map = dict(atoms)
    clips = [_build_one_topic_clip(t, atom_map, depth=0) for t in topics_data]
    collapse_trivial_children(clips, atom_map)
    deduplicate_parent_refs(clips, atom_map)
    return clips


# ── simplification: collapse trivial children ────────────────────────


def _try_collapse_overlapping_children(
    clip: TopicClip,
    atom_map: dict[int, str],
) -> bool:
    """Case 1: children's block_refs overlap (over-splitting) — absorb into parent."""
    all_child_refs: list[int] = []
    for child in clip.children:
        all_child_refs.extend(child.block_refs)
    if len(set(all_child_refs)) >= len(clip.children):
        return False
    merged = sorted(set(clip.block_refs) | set(all_child_refs))
    clip.block_refs = merged
    clip.content = _rebuild_content(merged, atom_map)
    clip.children = []
    return True


def _try_collapse_heading_only_parent(
    clip: TopicClip,
    atom_map: dict[int, str],
) -> None:
    """Case 2: parent is heading-only (<= 1 block) with a single child — promote child."""
    if len(clip.children) != 1 or len(clip.block_refs) > 1:
        return
    child = clip.children[0]
    merged = sorted(set(clip.block_refs) | set(child.block_refs))
    clip.topic = child.topic
    clip.summary = child.summary
    clip.block_refs = merged
    clip.content = _rebuild_content(merged, atom_map)
    clip.children = child.children


def collapse_trivial_children(clips: list[TopicClip], atom_map: dict[int, str]) -> None:
    """Merge children back into parent when the split adds no value.

    Two cases handled bottom-up:

    1. All children share blocks (over-splitting) → absorb into parent.
    2. Single child + parent has <= 1 block (heading-only parent) → promote child.

    Mutates clips in-place.
    """
    for clip in clips:
        if not clip.children:
            continue
        collapse_trivial_children(clip.children, atom_map)
        if _try_collapse_overlapping_children(clip, atom_map):
            continue
        _try_collapse_heading_only_parent(clip, atom_map)


# ── simplification: deduplicate parent refs ──────────────────────────


def _collect_subtree_refs(children: list[TopicClip]) -> set[int]:
    refs: set[int] = set()
    for child in children:
        refs.update(child.block_refs)
        if child.children:
            refs.update(_collect_subtree_refs(child.children))
    return refs


def deduplicate_parent_refs(clips: list[TopicClip], atom_map: dict[int, str]) -> None:
    """Remove block_refs from a parent that already appear in any descendant.

    Mutates clips in-place. After removing refs, rebuilds content from the
    remaining block_refs.
    """
    for clip in clips:
        if not clip.children:
            continue
        child_refs = _collect_subtree_refs(clip.children)
        original_len = len(clip.block_refs)
        clip.block_refs = [r for r in clip.block_refs if r not in child_refs]
        if len(clip.block_refs) < original_len:
            clip.content = _rebuild_content(clip.block_refs, atom_map)
        deduplicate_parent_refs(clip.children, atom_map)


# ── coverage helper for post-processing ──────────────────────────────


def collect_all_block_refs(topics_data: list[dict[str, Any]]) -> set[int]:
    """Collect block_refs from all levels of a raw LLM ``topics`` list.

    Used by post-processing to detect uncovered blocks before the LLM
    leaf-split / orphan-assign passes.
    """
    covered: set[int] = set()
    for t in topics_data:
        covered.update(parse_segment_ranges(t.get("block_refs", "")))
        for child in t.get("children", []):
            covered.update(collect_all_block_refs([child]))
    return covered
