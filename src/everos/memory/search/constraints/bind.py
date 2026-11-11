"""The constraint block's optional model layer, ``search.constraint_bind`` (8.4).

``[search] constraints = "model"`` adds one call with the extraction model
after the rules layer: given the question, the core and the rule
constraints, it states for each constraint whether it bears on the question
and to whom it applies (the group, or some members), citing an exact quote
from the core's segments. Code checks every binding and drops -- never
repairs -- one that fails:

* ``unknown_constraint``: it names no constraint of the block, or one a
  previous binding already bound (a duplicate);
* ``unknown_message`` / ``quote_not_verbatim``: the cited message is not in
  the named segment (one of the constraint's or the core's), or the quote is
  not a verbatim part of it (:func:`everos.memory.groups.verbatim_part`:
  whitespace normalised, curly quotation marks read as straight ones);
* ``unknown_member``: ``applies_to`` names someone not in the group's
  roster, or a binding that applies leaves it empty.

A binding that applies cites a message and states its scope. One that does
not (``applies: false``) may leave both empty -- no message, no quote, an
empty ``applies_to`` -- and then keeps the constraint's rules-layer source
and scope; whatever it does fill in is checked in full. One member id where
a list is expected reads as a list of it. A bound constraint gains
``applies`` and the binding's scope, and its quote is the message's own
words.

A constraint without a surviving binding keeps what the rules layer gave it,
and so does every constraint when the call itself yields nothing usable (an
invalid reply, a failed call): the search answers with the rules layer's
block, and the failure is counted in
``everos_constraint_bind_failures_total`` by reason. Dropped bindings are
counted in ``everos_constraint_bindings_dropped_total``. The call is
zero-shot and names the conversation's language, resolved in code from the
cited segments; the prompt is registered with an id, a version and a
fingerprint, pinned by tests.
"""

from __future__ import annotations

__all__ = ["CONSTRAINT_BIND", "BindReply", "Binding", "bind_constraints"]

import json
from collections import Counter
from collections.abc import Mapping, Sequence

from everalgo.llm import LLMClient, LLMMessage, StrictOutputModel, parse_reply
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    build_language_rule,
)
from pydantic import field_validator

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.memory.access import ReadScope
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.llm_retry import (
    UNUSABLE_CALL,
    failure_reason,
    retry_llm_output,
)
from everos.memory.groups import normalise_space, verbatim_part
from everos.memory.prompts import LENIENT_REPLY, PromptTemplate

from ..dto import ConstraintItem, ConstraintScopeItem, ConstraintSourceItem

logger = get_logger(__name__)

CONSTRAINT_BIND = PromptTemplate(
    id="search.constraint_bind",
    version=1,
    parts=(
        """You check which constraints an answer to a question must respect. \
The constraints come from a group chat's memory: decisions the members made, \
disagreements, topic states, and what was said in the relevant parts of the \
conversation. For each constraint, say whether it bears on the question and \
to whom it applies, and support that with words copied from the conversation.

<rules>
1. Answer once per constraint id you are given, in the given order.
2. "applies" is true when the answer must respect the constraint, false when \
the constraint is beside the point of the question.
3. "applies_to" is "group" when the constraint holds for the whole group, \
otherwise the list of member ids it holds for, taken from <members>.
4. "message" is the id in square brackets at the start of a conversation \
line; "quote" is words copied exactly from that line's text -- not its time or \
speaker. Copy; never paraphrase, shorten inside, correct or translate.
5. {language_rule}
</rules>

Reply with JSON only, in this shape:
{{"bindings": [{{"constraint": "c1", "applies": true, "applies_to": "group", \
"segment": "", "message": "", "quote": ""}}]}}""",
        """<question>
{question}
</question>

<members>
{members}
</members>

<constraints>
{constraints}
</constraints>

<conversation>
{conversation}
</conversation>""",
    ),
)
"""System and user text of the binding call."""

_dropped = MetricCounter(
    name="constraint_bindings_dropped_total",
    description="Constraint bindings from the model layer dropped by the verifier.",
    labelnames=("reason",),
    namespace="everos",
)
_failed = MetricCounter(
    name="constraint_bind_failures_total",
    description=(
        "Constraint-binding calls that yielded nothing usable; "
        "the block kept its rules form."
    ),
    labelnames=("reason",),
    namespace="everos",
)


class Binding(StrictOutputModel):
    """One constraint's binding as the model states it.

    The citation reads as empty when its keys are left out or ``null``: a
    binding that does not apply may carry none.
    """

    model_config = LENIENT_REPLY

    constraint: str
    applies: bool
    applies_to: str | list[str]
    segment: str = ""
    message: str = ""
    quote: str = ""

    @field_validator("segment", "message", "quote", mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return "" if value is None else value


class BindReply(StrictOutputModel):
    bindings: list[Binding]


async def bind_constraints(
    llm: LLMClient,
    *,
    question: str,
    scope: ReadScope,
    constraints: Sequence[ConstraintItem],
    cells: Mapping[str, MemCell],
) -> tuple[list[ConstraintItem], Counter[str]]:
    """The constraints with the bindings that survived the checks; drop counts.

    ``cells`` are the segments the constraints and the core rest on, by
    memcell id, their speakers labelled by roster name: what the call is
    shown and what every quote is checked against. A call that yields
    nothing usable leaves ``constraints`` as they are.
    """
    if not constraints:
        return [], Counter()
    language = conversation_language(list(cells.values()))
    system, user = CONSTRAINT_BIND.parts
    messages = [
        LLMMessage(
            role="system",
            content=system.format(
                language_rule=build_language_rule(
                    language, fallback=PARTICIPANT_LANGUAGE_RULE
                )
            ),
        ),
        LLMMessage(
            role="user",
            content=user.format(
                question=question,
                members=_members(scope),
                constraints="\n".join(_render(c) for c in constraints),
                conversation=_conversation(cells),
            ),
        ),
    ]

    async def attempt() -> BindReply:
        text = await llm.achat(messages, response_format=BindReply)
        return parse_reply(BindReply, text, operator="constraint binding")

    try:
        reply = await retry_llm_output(attempt, operator="constraint_bind")
    except UNUSABLE_CALL as err:
        # Only a refinement failed: the rules layer's block stands.
        reason = failure_reason(err)
        _failed.labels(reason=reason).inc()
        logger.warning(
            "constraint_bind_failed", reason=reason, constraints=len(constraints)
        )
        return list(constraints), Counter()
    return _apply(reply.bindings, constraints, scope=scope, cells=cells)


def _apply(
    bindings: Sequence[Binding],
    constraints: Sequence[ConstraintItem],
    *,
    scope: ReadScope,
    cells: Mapping[str, MemCell],
) -> tuple[list[ConstraintItem], Counter[str]]:
    by_id = {c.id: c for c in constraints}
    bound: dict[str, ConstraintItem] = {}
    dropped: Counter[str] = Counter()
    for binding in bindings:
        reason, updated = _check(binding, by_id, bound, scope=scope, cells=cells)
        if reason is not None:
            dropped[reason] += 1
            _dropped.labels(reason=reason).inc()
            continue
        assert updated is not None
        bound[updated.id] = updated
    if dropped:
        logger.info("constraint_bindings_dropped", dropped=dict(dropped))
    return [bound.get(c.id, c) for c in constraints], dropped


def _check(
    binding: Binding,
    by_id: Mapping[str, ConstraintItem],
    bound: Mapping[str, ConstraintItem],
    *,
    scope: ReadScope,
    cells: Mapping[str, MemCell],
) -> tuple[str | None, ConstraintItem | None]:
    constraint = by_id.get(binding.constraint)
    if constraint is None or binding.constraint in bound:
        return "unknown_constraint", None
    source = constraint.source
    if binding.applies or binding.message.strip() or binding.quote.strip():
        allowed = {constraint.source.memcell_id, *cells}
        cell = cells.get(binding.segment) if binding.segment in allowed else None
        message = next(
            (
                m
                for m in (cell.items if cell is not None else [])
                if isinstance(m, ChatMessage) and m.id == binding.message
            ),
            None,
        )
        if message is None:
            return "unknown_message", None
        quote = verbatim_part(binding.quote, str(message.content or ""))
        if quote is None:
            return "quote_not_verbatim", None
        source = ConstraintSourceItem(
            memcell_id=binding.segment,
            message_id=message.id,
            speaker_id=message.sender_id,
            at=message.timestamp,
            quote=quote,
        )
    applies_to = _applies_to(binding, constraint, scope)
    if applies_to is None:
        return "unknown_member", None
    return None, constraint.model_copy(
        update={
            "applies": binding.applies,
            "scope": ConstraintScopeItem(
                group_id=constraint.scope.group_id, applies_to=applies_to
            ),
            "source": source,
        }
    )


def _applies_to(
    binding: Binding, constraint: ConstraintItem, scope: ReadScope
) -> str | list[str] | None:
    """The binding's scope; ``None`` when it names an unknown member or none.

    ``"group"`` is the whole group, and one id reads as a list of it. A
    binding that does not apply may leave the scope empty: the constraint's
    own scope stands.
    """
    value = binding.applies_to
    if value == "group":
        return "group"
    if isinstance(value, str):
        members = [value] if value.strip() else []
    else:
        members = list(value)
    if not members:
        return None if binding.applies else constraint.scope.applies_to
    view = scope.group(constraint.scope.group_id)
    known = {i.member_id for i in view.roster.intervals} if view else set()
    if any(m not in known for m in members):
        return None
    return list(dict.fromkeys(members))


def _render(constraint: ConstraintItem) -> str:
    return json.dumps(
        {
            "id": constraint.id,
            "kind": constraint.kind,
            "status": constraint.status,
            "statement": constraint.statement,
            "group": constraint.scope.group_id,
            "quote": constraint.source.quote,
        },
        ensure_ascii=False,
    )


def _members(scope: ReadScope) -> str:
    lines = [
        f"- {i.member_id}: {i.display_name} (group {view.group_id})"
        for view in scope.groups
        for i in view.roster.current_members()
    ]
    return "\n".join(lines) or "(none)"


def _conversation(cells: Mapping[str, MemCell]) -> str:
    blocks: list[str] = []
    for memcell_id, cell in cells.items():
        lines = [f"=== SEGMENT {memcell_id} ==="]
        for item in cell.items:
            if isinstance(item, ChatMessage):
                text = normalise_space(str(item.content or ""))
                lines.append(
                    f"[{item.id}] {item.sender_name or item.sender_id}: {text}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
