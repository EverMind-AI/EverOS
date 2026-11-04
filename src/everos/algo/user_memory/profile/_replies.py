"""Each stage's reply model, the shape its prompt's output skeleton shows, and the LLM callsites held to it.

Each callsite makes one call and validates the reply against its stage's output model; there is no retry here,
the host decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from everos.algo.llm import LLMMessage, LLMOutputError, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _ExplicitItem(StrictOutputModel):
    """An ``explicit_info`` item as every stage writes it, grounding first: ``{"evidence", "description", "category"}``."""

    evidence: str
    description: str
    category: str


class _ImplicitTrait(StrictOutputModel):
    """An ``implicit_traits`` item as every stage writes it, grounding first: ``{"basis", "description", "trait"}``."""

    basis: str
    description: str
    trait: str


class _ProfileItems(StrictOutputModel):
    """INIT reply: the ``{"explicit_info", "implicit_traits"}`` skeleton of both INIT prompts."""

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]


class _CompactOutput(StrictOutputModel):
    """COMPACT reply on the Episode path: the rewritten buckets, nothing else."""

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]


class _MemCellCompactOutput(StrictOutputModel):
    """COMPACT reply on the MemCell path: the rewritten buckets plus the ``compact_note`` its prompt asks for."""

    explicit_info: list[_ExplicitItem]
    implicit_traits: list[_ImplicitTrait]
    compact_note: str


class _RegroupOutput(StrictOutputModel):
    """REGROUP reply on the Episode path: the group's kept items, nothing else."""

    items: list[_ExplicitItem | _ImplicitTrait]


class _MemCellRegroupOutput(StrictOutputModel):
    """REGROUP reply on the MemCell path: the group's kept items plus the ``regroup_note`` its prompt asks for."""

    items: list[_ExplicitItem | _ImplicitTrait]
    regroup_note: str


_Bucket = Literal["explicit_info", "implicit_traits"]
# The item shape each bucket holds; the business checks after parsing hold replies to it.
_BUCKET_SHAPES: dict[str, type[_ExplicitItem | _ImplicitTrait]] = {
    "explicit_info": _ExplicitItem,
    "implicit_traits": _ImplicitTrait,
}


class _AddOp(StrictOutputModel):
    """UPDATE ``add`` operation: a new item for the bucket named by its ``type`` key."""

    action: Literal["add"]
    # The JSON key stays "type", as the UPDATE prompts show it; the field name avoids the builtin.
    bucket: _Bucket = Field(alias="type")
    data: _ExplicitItem | _ImplicitTrait


class _UpdateOp(StrictOutputModel):
    """UPDATE ``update`` operation: the item at ``index`` restated in full."""

    action: Literal["update"]
    bucket: _Bucket = Field(alias="type")
    index: int
    data: _ExplicitItem | _ImplicitTrait


class _DeleteOp(StrictOutputModel):
    """UPDATE ``delete`` operation: the item at ``index`` removed, with the model's ``reason``."""

    action: Literal["delete"]
    bucket: _Bucket = Field(alias="type")
    index: int
    reason: str


class _NoneOp(StrictOutputModel):
    """UPDATE ``none`` operation: the documented no-op."""

    action: Literal["none"]


_Operation = Annotated[
    _AddOp | _UpdateOp | _DeleteOp | _NoneOp, Field(discriminator="action")
]


class _UpdateOutput(StrictOutputModel):
    """UPDATE reply on the Episode path: the ``{"operations"}`` skeleton of its prompt."""

    operations: list[_Operation]


class _MemCellUpdateOutput(StrictOutputModel):
    """UPDATE reply on the MemCell path: the operations plus the ``update_note`` its prompt asks for."""

    operations: list[_Operation]
    update_note: str


# Each path validates against its own models, repeated rather than extended.
_AnyUpdateOutput = _UpdateOutput | _MemCellUpdateOutput
_AnyCompactOutput = _CompactOutput | _MemCellCompactOutput
_AnyRegroupOutput = _RegroupOutput | _MemCellRegroupOutput


async def _call_llm_for_profile_init(
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[_ProfileItems]
) -> dict[str, Any]:
    """Request an INIT reply and return it validated, dumped to a dict.

    Raises:
        LLMOutputError: If the reply is not a valid INIT output.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    return parse_reply(reply, text, operator="profile init").model_dump(by_alias=True)


async def _call_llm_for_profile_update[ReplyT: _AnyUpdateOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT]
) -> dict[str, Any]:
    """Request an UPDATE reply and return it validated, dumped to a dict whose ``operations`` keep their ``type`` key.

    Raises:
        LLMOutputError: If the reply is not a valid UPDATE output, or an add or update carries an item not shaped
            for the bucket its ``type`` names.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    output = parse_reply(reply, text, operator="profile update")
    _check_bucket_shapes(output)
    return output.model_dump(by_alias=True)


async def _call_llm_for_profile_compact[ReplyT: _AnyCompactOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT]
) -> dict[str, Any]:
    """Request a COMPACT reply and return it validated, dumped to a dict.

    Raises:
        LLMOutputError: If the reply is not a valid COMPACT output.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    return parse_reply(reply, text, operator="profile compact").model_dump(
        by_alias=True
    )


async def _call_llm_for_profile_regroup[ReplyT: _AnyRegroupOutput](
    llm: LLMClient, messages: list[LLMMessage], *, reply: type[ReplyT], bucket: str
) -> dict[str, Any]:
    """Request a REGROUP reply for a group of ``bucket`` and return it validated, dumped to a dict.

    Raises:
        LLMOutputError: If the reply is not a valid REGROUP output, or an item is not shaped for ``bucket``.
    """
    text = await llm.achat(messages=messages, response_format=reply)
    output = parse_reply(reply, text, operator="profile regroup")
    _check_item_shapes(output, bucket=bucket)
    return output.model_dump(by_alias=True)


def _check_bucket_shapes(output: _AnyUpdateOutput) -> None:
    """Reject an add or update whose item is not shaped for the bucket its ``type`` names (llm-contract §4.1).

    Raises:
        LLMOutputError: Naming each mismatched operation by position, its bucket and the shape it carried.
    """
    mismatches = [
        f"operations.{position}: {op.action} into {op.bucket} carries a {type(op.data).__name__}"
        for position, op in enumerate(output.operations)
        if isinstance(op, _AddOp | _UpdateOp)
        and not isinstance(op.data, _BUCKET_SHAPES[op.bucket])
    ]
    if mismatches:
        raise LLMOutputError(
            f"profile update reply puts items in the wrong bucket: {'; '.join(mismatches)}"
        )


def _check_item_shapes(output: _AnyRegroupOutput, *, bucket: str) -> None:
    """Reject a regrouped item that is not shaped for the group's bucket; regroup never moves an item across.

    Raises:
        LLMOutputError: Naming each mismatched item by position and the shape it carried.
    """
    expected = _BUCKET_SHAPES[bucket]
    mismatches = [
        f"items.{position}: a {type(item).__name__}"
        for position, item in enumerate(output.items)
        if not isinstance(item, expected)
    ]
    if mismatches:
        raise LLMOutputError(
            f"profile regroup reply for a {bucket} group holds items not shaped as {expected.__name__}: "
            f"{'; '.join(mismatches)}"
        )
