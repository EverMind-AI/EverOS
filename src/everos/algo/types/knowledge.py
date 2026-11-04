"""Knowledge memory contract — flattened topic-tree node emitted by KnowledgeExtractor."""

from __future__ import annotations

__all__ = ["CategorySpec", "KnowledgeMemory"]

from pydantic import BaseModel, Field


class KnowledgeMemory(BaseModel):
    """A single topic-tree node from a parsed document, in flattened (DFS) form.

    Output unit of ``KnowledgeExtractor.aextract``. Each ``aextract`` call
    returns ``list[KnowledgeMemory]`` for one document; the first element
    (``topic_index=0``, ``depth=0``) is the document root node.

    Root node convention
    --------------------
    For every document the entry at ``topic_index=0`` is constructed
    programmatically (not LLM-extracted) and carries document-level
    context:

    * ``topic = document title`` — exempt from the ≤ 20 chars rule that
      applies to LLM-extracted section names.
    * ``summary = document-level summary``.
    * ``content`` is usually empty (container node).
    * ``parent_index = None``, ``depth = 0``.

    This places document-level signal directly inside the node list,
    so downstream cross-document operators do not require a separate
    ``doc_meta_lookup`` argument.

    Identity
    --------
    The composite key is ``(doc_id, topic_index)``.
    """

    doc_id: str = Field(description="Identifier of the source document.")
    topic_index: int = Field(
        description="DFS order within the document; the root node is index 0.",
    )
    topic: str = Field(
        description=(
            "Section / topic name. LLM-extracted nodes follow a ≤ 20 chars "
            "soft cap; the root node carries the full document title and is exempt."
        ),
    )
    summary: str = Field(description="1-2 sentence summary of this node.")
    content: str = Field(
        default="",
        description=(
            "Original prose under this section, joined from atoms via ``block_refs``. Empty for container nodes."
        ),
    )
    block_refs: list[int] = Field(
        default=[],
        description="Indices into the document's atom list that compose this node.",
    )
    depth: int = Field(description="Tree depth; 0 is the document root.")
    parent_index: int | None = Field(
        default=None,
        description="``topic_index`` of the parent node; ``None`` for the root.",
    )
    children_index: list[int] = Field(
        default=[],
        description="``topic_index`` values of the direct children, in DFS order.",
    )
    topic_path: str = Field(
        description="Human-readable hierarchy trail, e.g. ``A > A.1 > A.1.1``.",
    )
    content_labels: list[str] = Field(
        default_factory=list,
        description="Free-form labels propagated from the document (safety / privacy / topical).",
    )
    category_id: str = Field(
        default="",
        description=(
            "Document-level category id, denormalized onto every node of the same document. "
            "Empty string means unclassified (default when the extractor was called without a "
            "``categories=`` taxonomy)."
        ),
    )


class CategorySpec(BaseModel):
    """One entry in a caller-owned classification taxonomy.

    The taxonomy is business data and lives in the caller — the algorithms
    ships this type so the contract is shared, but never bundles a default taxonomy.
    A typical deployment passes ~30 of these into ``KnowledgeExtractor.aextract`` and
    the resulting ``id`` is stamped onto every emitted ``KnowledgeMemory`` of the
    document.
    """

    id: str = Field(
        description="Stable identifier written into ``KnowledgeMemory.category_id``."
    )
    description: str = Field(
        description=(
            "Short (~100 token) prose describing the kinds of documents this category fits. "
            "Used verbatim as the LLM classifier's taxonomy prefix; should be stable across "
            "ingest runs so prompt caching takes effect."
        ),
    )
