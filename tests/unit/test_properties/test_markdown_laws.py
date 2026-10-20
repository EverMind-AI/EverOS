"""Round-trip laws of the markdown layer, checked on generated documents.

Markdown is the source of truth: whatever a writer renders, a reader (the
cascade handlers, the CLI, a human's editor) must parse back to the same
thing, or the index silently drifts from the files. Each law below states
one writer/reader pair as "parse after render is the identity" over a
generated domain, with the domain's limits spelled out:

1. ``EntryId.parse(EntryId.format(e)) == e`` for any prefix (including a
   multi-segment one), any date from year 1000 and any non-negative seq.
2. ``dump_frontmatter`` then ``parse_frontmatter`` returns the mapping and
   the body untouched, for a non-empty mapping of single-line scalars,
   lists and one level of nesting. Two limits are the format's own, not
   bugs: an empty mapping renders as no block at all, so a body that itself
   starts with ``---`` would then be read as frontmatter; and a value
   containing a line that is exactly ``---`` would close the block early.
3. ``split_entries`` recovers every ``<!-- entry:id -->`` span in order with
   its body, and ``find_entry`` agrees with it, for bodies that contain no
   marker text and do not end in a bare ``\r``: the reader strips one
   newline of padding on each side of a span, and a trailing ``\r`` glued
   to the padding's ``\n`` reads as one ``\r\n`` newline. Entry bodies
   end in ``\n`` or in text, never in a bare carriage return.
4. ``render_structured_entry`` then ``parse_structured_entry`` returns the
   header, the inline fields and the sections, for single-line keys and
   values and section bodies with no ``###`` heading line. The audit form
   has no escaping, so a section body that itself contains a ``### ``
   line splits into two sections: a documented limit of the format.
5. ``parse_inline_list`` inverts ``_render_value`` for lists of ids drawn
   from the entry-id alphabet (no commas, no edge whitespace).
"""

from __future__ import annotations

import datetime as dt

from hypothesis import given
from hypothesis import strategies as st

from everos.core.persistence.markdown import entries as ent
from everos.core.persistence.markdown import frontmatter as fm
from everos.memory.cascade.handlers._common import parse_inline_list

_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
_line = st.text(st.characters(exclude_categories=("Cc", "Cs", "Zl", "Zp")), max_size=40)
_stripped_line = _line.map(str.strip).filter(bool)
_ident = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)
_scalar = st.one_of(_line, st.integers(-(10**9), 10**9), st.booleans())
_meta = st.dictionaries(
    _ident,
    st.one_of(
        _scalar,
        st.lists(_line, max_size=4),
        st.dictionaries(_ident, _scalar, max_size=3),
    ),
    min_size=1,
    max_size=6,
)
_body = st.text(max_size=200)


@given(
    prefix=st.from_regex(r"[A-Za-z0-9-]+(_[A-Za-z0-9-]+)*", fullmatch=True),
    date=st.dates(min_value=dt.date(1000, 1, 1)),
    seq=st.integers(0, 10**12),
)
def test_entry_id_round_trips(prefix: str, date: dt.date, seq: int) -> None:
    e = ent.EntryId(prefix=prefix, date=date, seq=seq)
    assert ent.EntryId.parse(e.format()) == e
    assert ent.EntryId.parse(str(e)) == e


@given(meta=_meta, body=_body)
def test_frontmatter_round_trips(meta: dict, body: str) -> None:
    text = fm.dump_frontmatter(meta) + body
    assert fm.parse_frontmatter(text) == (meta, body)


def _render_entries(items: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"<!-- entry:{eid} -->\n{body}\n<!-- /entry:{eid} -->" for eid, body in items
    )


@given(
    items=st.lists(
        st.tuples(
            st.text(_ID_ALPHABET, min_size=1, max_size=24),
            st.text(max_size=120).filter(
                lambda s: "<!--" not in s and not s.endswith("\r")
            ),
        ),
        max_size=6,
        unique_by=lambda item: item[0],
    )
)
def test_split_and_find_recover_every_entry(items: list[tuple[str, str]]) -> None:
    doc = _render_entries(items)
    got = [(e.id, e.body) for e in ent.split_entries(doc)]
    assert got == items
    for eid, body in items:
        found = ent.find_entry(doc, eid)
        assert found is not None and found.body == body


_section_body = st.text(max_size=120).filter(
    lambda s: (
        not s.startswith("\n")
        and not any(line.startswith("###") for line in s.splitlines())
    )
)


@given(
    header=st.one_of(st.none(), _stripped_line),
    inline=st.dictionaries(
        _stripped_line.filter(lambda k: "*" not in k and ":" not in k),
        _stripped_line,
        max_size=5,
    ),
    sections=st.dictionaries(_stripped_line, _section_body, max_size=4),
)
def test_structured_entry_round_trips(
    header: str | None, inline: dict[str, str], sections: dict[str, str]
) -> None:
    rendered = ent.render_structured_entry(
        header=header, inline=inline, sections=sections
    )
    parsed = ent.parse_structured_entry(rendered)
    assert parsed.header == header
    assert parsed.inline == inline
    assert parsed.sections == {t: b.rstrip() for t, b in sections.items()}


@given(ids=st.lists(st.text(_ID_ALPHABET, min_size=1, max_size=24), max_size=8))
def test_inline_list_round_trips(ids: list[str]) -> None:
    assert parse_inline_list(ent._render_value(ids)) == ids
