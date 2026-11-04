"""The knowledge reply models, the shapes their prompts' output skeletons show, and the one LLM callsite they share.

Each model declares its fields in its skeleton's order: a schema-constrained reply is generated in schema order. The
callsite makes one call and validates the reply; there is no retry here, the host decides.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

__all__ = [
    "Assignment",
    "CategoryReply",
    "ContentMergeReply",
    "MergeGroup",
    "SplitChild",
    "SplitResult",
    "TopicMergeReply",
    "TopicNode",
    "TopicSplitReply",
    "TopicTreeReply",
    "UncoveredAssignReply",
    "ask_llm",
]

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class TopicNode(StrictOutputModel):
    """One topic of the topic-tree skeleton; a leaf has ``children == []``."""

    topic: str
    summary: str
    block_refs: str
    content_labels: list[str]
    children: list[TopicNode]


class TopicTreeReply(StrictOutputModel):
    """Topic-tree extraction reply: the document-level fields, then the topic forest."""

    language: str
    summary: str
    subject: str
    keywords: list[str]
    content_labels: list[str]
    topics: list[TopicNode]


class ContentMergeReply(StrictOutputModel):
    """Cross-batch content-merge reply."""

    language: str
    summary: str
    subject: str
    keywords: list[str]


class MergeGroup(StrictOutputModel):
    """One group of duplicate topics, by index into the flattened topic list."""

    indices: list[int]
    topic: str
    summary: str


class TopicMergeReply(StrictOutputModel):
    """Cross-batch topic-merge reply; ``merges == []`` when nothing is a duplicate."""

    merges: list[MergeGroup]


class CategoryReply(StrictOutputModel):
    """Classification reply; ``""`` when no category fits."""

    category_id: str


class SplitChild(StrictOutputModel):
    """One child of a split leaf."""

    topic: str
    summary: str
    block_refs: str
    children: list[SplitChild]


class SplitResult(StrictOutputModel):
    """The split decision for one leaf; ``block_refs`` and ``children`` are null when ``should_split`` is false."""

    original_topic: str
    should_split: bool
    reason: str
    block_refs: str | None
    children: list[SplitChild] | None


class TopicSplitReply(StrictOutputModel):
    """Leaf-split reply: one result per leaf shown."""

    results: list[SplitResult]


class Assignment(StrictOutputModel):
    """One orphan block attached to a topic, by index into the topic list shown."""

    block_id: int
    topic_index: int
    reason: str


class UncoveredAssignReply(StrictOutputModel):
    """Uncovered-block assignment reply."""

    assignments: list[Assignment]


async def ask_llm[ReplyT: StrictOutputModel](
    llm: LLMClient,
    *,
    system: str,
    user: str,
    reply_model: type[ReplyT],
    operator: str,
) -> ReplyT:
    """Send the ``[system, user]`` pair, requesting ``reply_model``, and return the reply validated against it.

    Args:
        llm: The client every call goes through.
        system: The system message: the rules.
        user: The user message: the data.
        reply_model: The ``response_format`` model the reply is held to.
        operator: Names the call in the error message, e.g. ``"knowledge topic_tree_extract"``.

    Returns:
        The validated reply.

    Raises:
        LLMOutputError: If the reply is not valid JSON for ``reply_model``.
    """
    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]
    text = await llm.achat(messages=messages, response_format=reply_model)
    return parse_reply(reply_model, text, operator=operator)
