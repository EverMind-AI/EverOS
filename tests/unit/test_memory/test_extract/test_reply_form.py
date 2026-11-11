"""Mending a JSON reply's form without touching what it says.

What these pin (``everos.memory.extract.reply_form``):

- an array element left open is closed where the next token shows it ended,
  before a comma and before the array's ``]`` -- the slip DeepSeek V4.1 Flash
  made on the picture route's profile operations (``"data": {...}`` followed
  by ``, {`` or ``]`` where ``}}`` was due);
- a reply split into consecutive objects is joined when no key repeats;
- nothing else is mended: two braces missing, an unclosed object that is not
  an array element, an unterminated string, a key given twice across a split,
  a value left open at the end all return ``None``, and so does a text that
  needed nothing;
- an unknown key reads as absent only while its value is empty.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from everos.memory.extract.reply_form import (
    drop_empty_unknown_keys,
    is_empty,
    repair_json_form,
)


def test_an_element_left_open_before_the_next_is_closed() -> None:
    text = '{"ops": [{"a": 1, "data": {"x": 2}, {"a": 3, "data": {"x": 4}}]}'

    assert json.loads(repair_json_form(text) or "") == {
        "ops": [{"a": 1, "data": {"x": 2}}, {"a": 3, "data": {"x": 4}}]
    }


def test_an_element_left_open_before_the_array_closes_is_closed() -> None:
    text = '{"ops": [{"a": 1, "data": {"x": 2}]}'

    assert json.loads(repair_json_form(text) or "") == {
        "ops": [{"a": 1, "data": {"x": 2}}]
    }


def test_every_element_of_an_array_may_be_left_open() -> None:
    text = '{"ops": [{"a": 1, "data": {"x": 2}, {"a": 3, "data": {"x": 4}]}'

    assert json.loads(repair_json_form(text) or "") == {
        "ops": [{"a": 1, "data": {"x": 2}}, {"a": 3, "data": {"x": 4}}]
    }


def test_braces_inside_strings_are_text_not_structure() -> None:
    text = '{"ops": [{"q": "a \\" } ] {, [", "data": {"x": "}"}]}'

    assert json.loads(repair_json_form(text) or "") == {
        "ops": [{"q": 'a " } ] {, [', "data": {"x": "}"}}]
    }


def test_a_reply_split_in_two_objects_is_joined() -> None:
    text = '{"content": "c", "images": [{"ref": "img-1"}]}\n{"profile": []}'

    assert json.loads(repair_json_form(text) or "") == {
        "content": "c",
        "images": [{"ref": "img-1"}],
        "profile": [],
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',  # nothing to mend
        '{"ops": [{"a": 1, "data": {"x": 2, {"a": 3}]}',  # two braces missing
        '{"a": {"b": 1}, {"c": 2}}',  # the open object is not an array element
        '{"a": "never closed',
        '{"a": [1, 2',
        '{"a": 1}\n{"a": 2}',  # a key given twice across the split
        '{"a": 1}\n[2]',  # a split whose second part is not an object
        '{"ops": [{"a": 1}]}]}',  # a bracket too many
        '{"a" 1}',
        "",
    ],
)
def test_any_other_slip_is_left_to_fail(text: str) -> None:
    assert repair_json_form(text) is None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    kind: str = Field(alias="type")


@pytest.mark.parametrize("empty", ["", "  ", None, [], {}])
def test_an_unknown_key_left_empty_reads_as_absent(empty: object) -> None:
    value = {"name": "n", "type": "t", "note": empty}

    assert drop_empty_unknown_keys(_Model, value) == {"name": "n", "type": "t"}


def test_an_unknown_key_with_content_is_kept_for_the_model_to_reject() -> None:
    value = {"name": "n", "type": "t", "note": "said something", "count": 0}

    assert drop_empty_unknown_keys(_Model, value) == value


def test_a_known_field_is_never_dropped_even_when_empty() -> None:
    value = {"name": "", "kind": None}

    assert drop_empty_unknown_keys(_Model, value) == value


def test_anything_but_a_mapping_passes_through() -> None:
    assert drop_empty_unknown_keys(_Model, ["a"]) == ["a"]


@pytest.mark.parametrize(
    ("value", "empty"),
    [
        (None, True),
        ("", True),
        (" \n", True),
        ([], True),
        ({}, True),
        ("x", False),
        (0, False),
        (False, False),
        ([""], False),
        ({"a": ""}, False),
    ],
)
def test_what_counts_as_empty(value: object, empty: bool) -> None:
    assert is_empty(value) is empty
