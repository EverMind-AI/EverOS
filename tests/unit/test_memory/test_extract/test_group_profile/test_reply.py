"""The group-profile reply: lenient about form, strict schema on the wire.

docs/group-memory.md, section 7.5 and the Step 2 notes of section 14. What
these pin:

- a key left out or ``null`` reads as empty, and one id where a list is
  expected as a list of it (``""`` as no one) -- the shapes DeepSeek V4.1
  Flash wrote in JSON mode in 33 of 42 replies of the 2026-11-11 check;
- what is not a form slip still fails: an unknown operation, a number for
  an id, an extra key;
- the JSON schema the call sends is the one without defaults: every field
  required, no ``default`` anywhere.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from everalgo.llm import to_openai_response_format
from pydantic import ValidationError

from everos.memory.extract.group_profile import GroupProfileReply


def _one(op: dict[str, Any]) -> Any:
    reply = GroupProfileReply.model_validate_json(json.dumps({"operations": [op]}))
    return reply.operations[0]


def test_left_out_and_null_fields_read_as_empty() -> None:
    op = _one({"op": "topic.new", "name": "Travel", "summary": None})

    assert (op.name, op.summary, op.target, op.statement) == ("Travel", "", "", "")
    assert op.decided_by == [] and op.held_by == [] and op.evidence == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("user_101", ["user_101"]), ("", []), (None, []), (["a", "b"], ["a", "b"])],
)
def test_one_id_reads_as_a_list(value: object, expected: list[str]) -> None:
    op = _one({"op": "decision.new", "decided_by": value, "agreed_by": value})

    assert op.decided_by == expected and op.agreed_by == expected


def test_an_evidence_entry_may_leave_keys_out_too() -> None:
    op = _one({"op": "topic.new", "evidence": [{"segment": "mc_1", "message": "m-1"}]})

    assert (op.evidence[0].segment, op.evidence[0].quote) == ("mc_1", "")


@pytest.mark.parametrize(
    "op",
    [
        {"op": "topic.rename"},
        {"op": "decision.new", "decided_by": [7]},
        {"op": "topic.new", "reason": "extra"},
        {"name": "no op"},
    ],
)
def test_what_is_not_a_form_slip_still_fails(op: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _one(op)


def test_the_schema_sent_is_strict_and_has_no_defaults() -> None:
    sent = to_openai_response_format(GroupProfileReply, "json_schema")
    assert sent is not None
    schema = sent["json_schema"]["schema"]
    text = json.dumps(schema)

    assert '"default"' not in text
    for model in schema["$defs"].values():
        assert model["required"] == list(model["properties"])
        assert model["additionalProperties"] is False
