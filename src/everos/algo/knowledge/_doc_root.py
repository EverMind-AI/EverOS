"""Document-root construction for ``KnowledgeExtractor``.

Pure-compute helper that wraps an extracted ``TopicClip`` forest under a
synthetic root ``TopicClip``. The root carries document-level fields the
LLM produced alongside the topic tree (title / summary / labels), so the
public flattened output can place document-level signal at index 0 with
``depth=0`` per the ``KnowledgeMemory`` root convention.

This step is *not* a separate LLM call — the source data comes from the
single combined topic-tree prompt; this module simply lifts those fields
into a tree node.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

from everos.algo.knowledge._topic_build import TopicClip

__all__ = ["build_doc_root"]


def build_doc_root(
    forest: list[TopicClip],
    *,
    doc_title: str,
    doc_summary: str,
    doc_content_labels: list[str] | None = None,
) -> TopicClip:
    """Wrap an extracted forest under a synthetic document root.

    Args:
        forest: The top-level TopicClip nodes returned by ``build_topic_clips``.
        doc_title: Document title, used verbatim as the root's ``topic``. The
            ``<=20`` chars cap on LLM-extracted section names does *not* apply —
            the root is programmatically constructed, not LLM-extracted.
        doc_summary: Document-level summary (e.g. from the LLM's ``summary`` field).
        doc_content_labels: Document-level content labels propagated to the root.
            Defaults to an empty list.

    Returns:
        A single root ``TopicClip`` whose ``children`` is the input forest. Root
        ``content`` is empty (container node) and ``block_refs`` is empty
        (the root does not own any atoms directly).
    """
    return TopicClip(
        topic=doc_title,
        summary=doc_summary,
        content="",
        block_refs=[],
        content_labels=list(doc_content_labels or []),
        children=list(forest),
    )
