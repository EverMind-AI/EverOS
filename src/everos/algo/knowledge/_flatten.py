"""DFS flatten ``TopicClip`` tree into ``list[KnowledgeMemory]``.

Pure-compute stage that converts the internal intermediate tree into
the public output shape of ``KnowledgeExtractor.aextract``. The root
node lands at ``topic_index=0`` with ``depth=0`` per the
``KnowledgeMemory`` root convention; descendants follow DFS order with
``topic_index`` assigned in visit order.

``topic_path`` is built by joining each visited node's ``topic`` with
`` > `` from the root down. ``children_index`` is back-filled in a
second pass from ``parent_index`` pointers.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everos.algo.types import KnowledgeMemory

if TYPE_CHECKING:
    from everos.algo.knowledge._topic_build import TopicClip

__all__ = ["flatten"]


def _visit(
    clip: TopicClip,
    *,
    doc_id: str,
    category_id: str,
    parent_index: int | None,
    depth: int,
    path_parts: list[str],
    out: list[dict[str, Any]],
) -> None:
    """Append one record per node in DFS order; children_index is left empty."""
    own_index = len(out)
    new_path_parts = [*path_parts, clip.topic]
    out.append(
        {
            "doc_id": doc_id,
            "topic_index": own_index,
            "topic": clip.topic,
            "summary": clip.summary,
            "content": clip.content,
            "block_refs": list(clip.block_refs),
            "depth": depth,
            "parent_index": parent_index,
            "children_index": [],
            "topic_path": " > ".join(new_path_parts),
            "content_labels": list(clip.content_labels),
            "category_id": category_id,
        },
    )
    for child in clip.children:
        _visit(
            child,
            doc_id=doc_id,
            category_id=category_id,
            parent_index=own_index,
            depth=depth + 1,
            path_parts=new_path_parts,
            out=out,
        )


def flatten(
    root: TopicClip, *, doc_id: str, category_id: str = ""
) -> list[KnowledgeMemory]:
    """Flatten a single-root ``TopicClip`` tree into ``list[KnowledgeMemory]``.

    Args:
        root: Single-root tree (typically produced by ``build_doc_root``). DFS
            from ``root`` produces records in left-to-right document order.
        doc_id: Identifier copied into every resulting ``KnowledgeMemory.doc_id``.
        category_id: Document-level category id (typically produced by
            ``aclassify_category``) denormalized onto every emitted node. Empty
            string means unclassified — the default when the extractor was called
            without a ``categories=`` taxonomy.

    Returns:
        ``list[KnowledgeMemory]`` — index 0 is the document root with ``depth=0``
        and ``parent_index=None``. Subsequent entries follow DFS order;
        ``topic_index`` equals the position in this list. ``children_index`` is
        populated by back-scanning ``parent_index`` pointers.
    """
    records: list[dict[str, Any]] = []
    _visit(
        root,
        doc_id=doc_id,
        category_id=category_id,
        parent_index=None,
        depth=0,
        path_parts=[],
        out=records,
    )

    # Backfill children_index from parent_index pointers (single linear pass).
    children_by_parent: dict[int, list[int]] = {}
    for rec in records:
        pidx = rec["parent_index"]
        if pidx is not None:
            children_by_parent.setdefault(pidx, []).append(rec["topic_index"])
    for rec in records:
        rec["children_index"] = children_by_parent.get(rec["topic_index"], [])

    return [KnowledgeMemory(**rec) for rec in records]
