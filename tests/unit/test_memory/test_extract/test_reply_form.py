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
- the lenient reading of a reply nothing can mend (the episode call's
  salvage): a string whose closing quote is lost ends at its line, a quote
  copied into a string unescaped is text, the other slips of form are read as
  meant, and where the text ends or stops being JSON the strings and
  containers left open are marked.
"""

from __future__ import annotations

import json

import pytest

from everos.memory.extract.reply_form import (
    CutText,
    OpenArray,
    OpenObject,
    read_leniently,
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


# ---------------------------------------------------------------------------
# read_leniently: as much of a broken reply as can be read, marked
# ---------------------------------------------------------------------------


def test_a_string_closed_by_a_typographic_quote_ends_at_its_line() -> None:
    """The closing quote was written as \u201d; the next line is the next key."""
    text = '{\n  "ocr": "PHASE ONE \u201cdefine\u201d\u201d\n  },\n  "next": "x"\n}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)

    value = read_leniently(text)

    assert isinstance(value, dict)
    assert value["ocr"] == "PHASE ONE \u201cdefine\u201d\u201d"
    assert isinstance(value["ocr"], CutText)
    assert value["ocr"].how == "line"


def test_a_quote_copied_into_a_string_unescaped_is_its_text() -> None:
    text = '{"ocr": "rating 2\'00" 1\'30" and "star" plan", "n": 1}'

    value = read_leniently(text)

    assert value == {"ocr": 'rating 2\'00" 1\'30" and "star" plan', "n": 1}
    assert not isinstance(value["ocr"], CutText)


def test_a_reply_cut_off_marks_what_it_was_reading() -> None:
    text = '{"title": "t", "images": [{"ref": "img-1", "ocr": "1 2 3 1 2 3 1'

    value = read_leniently(text)

    assert isinstance(value, OpenObject)
    assert value["title"] == "t"
    images = value["images"]
    assert isinstance(images, OpenArray)
    assert isinstance(images[0], OpenObject)
    assert images[0]["ref"] == "img-1"
    assert isinstance(images[0]["ocr"], CutText)
    assert images[0]["ocr"].how == "end"


def test_a_line_break_inside_a_string_followed_by_text_is_text() -> None:
    value = read_leniently('{"ocr": "line one\nline two", "n": 1}')

    assert value == {"ocr": "line one\nline two", "n": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a comma missing between members on separate lines
        ('{"a": "x"\n "b": "y"}', {"a": "x", "b": "y"}),
        # one comma too many
        ('{"a": "x",}', {"a": "x"}),
        # an array element left open, before the next one and before the end
        ('{"l": [{"a": 1, {"a": 2]}', {"l": [{"a": 1}, {"a": 2}]}),
        # an escape JSON does not know reads as the character escaped
        ('{"a": "it\\\'s"}', {"a": "it's"}),
        # a surrogate pair written as two escapes
        ('{"a": "\\ud83d\\ude00"}', {"a": "\U0001f600"}),
        # a reply split in two objects
        ('{"a": 1}\n{"b": 2}', {"a": 1, "b": 2}),
        # text around the object
        ('Here it is: {"a": 1} done', {"a": 1}),
    ],
)
def test_other_slips_of_form_are_read_as_meant(
    text: str, expected: dict[str, object]
) -> None:
    assert read_leniently(text) == expected


def test_a_text_with_no_object_reads_as_none() -> None:
    assert read_leniently("no reply at all") is None


def test_where_the_text_stops_being_json_reading_stops() -> None:
    value = read_leniently('{"a": "x", "b": ??? "c": "y"}')

    assert isinstance(value, OpenObject)
    assert value == {"a": "x"}
