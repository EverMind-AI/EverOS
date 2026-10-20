"""Laws of the backend-neutral predicate tree and its two renderers.

Every filter the search and delete paths send to LanceDB or Milvus is built
from ``everos.infra.persistence.predicate`` and rendered by one adapter. A
wrong rendering is a wrong result set, and on ``delete`` a wrong row set.

1. ``all_of`` / ``any_of`` flatten their own kind, drop ``None`` and reject
   an empty group, as do the ``All`` / ``AnyOf`` constructors themselves.
2. String literals survive each dialect's escaping: LanceDB doubles single
   quotes inside ``'...'``, Milvus uses a JSON string; both invert exactly.
3. Both renderers are total on well-formed trees (any depth, any scalar) and
   emit balanced parentheses; a single-child group renders as its child.
4. On a real LanceDB table, the rows a rendered filter selects are exactly
   the rows the predicate selects when evaluated in Python, for string,
   integer, float and boolean fields and arbitrary AND/OR nesting.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from everos.infra.persistence import predicate as pr
from everos.infra.persistence.lancedb.repos import _contract as lance
from everos.infra.persistence.milvus import predicate as milvus

_field = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,10}", fullmatch=True)
_text = st.text(max_size=30)
_scalar = st.one_of(
    _text,
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)


def _leaf(
    field_strategy: st.SearchStrategy[str], value_strategy: st.SearchStrategy[Any]
):
    return st.one_of(
        st.builds(pr.eq, field_strategy, value_strategy),
        st.builds(pr.ne, field_strategy, value_strategy),
        st.builds(pr.gt, field_strategy, value_strategy),
        st.builds(pr.lte, field_strategy, value_strategy),
        st.builds(
            pr.one_of, field_strategy, st.lists(value_strategy, min_size=1, max_size=3)
        ),
        st.builds(pr.is_null, field_strategy),
    )


def _trees(leaf: st.SearchStrategy[pr.Predicate]) -> st.SearchStrategy[pr.Predicate]:
    return st.recursive(
        leaf,
        lambda inner: st.one_of(
            st.lists(inner, min_size=1, max_size=3).map(lambda xs: pr.all_of(*xs)),
            st.lists(inner, min_size=1, max_size=3).map(lambda xs: pr.any_of(*xs)),
        ),
        max_leaves=8,
    )


any_tree = _trees(_leaf(_field, _scalar))


@given(xs=st.lists(st.one_of(st.none(), any_tree), max_size=5))
def test_all_of_and_any_of_flatten_drop_none_and_reject_empty(
    xs: list[pr.Predicate | None],
) -> None:
    present = [x for x in xs if x is not None]
    for factory, group in ((pr.all_of, pr.All), (pr.any_of, pr.AnyOf)):
        if not present:
            with pytest.raises(ValueError):
                factory(*xs)
            continue
        got = factory(*xs)
        assert isinstance(got, group)
        want: list[pr.Predicate] = []
        for x in present:
            want.extend(x.children if isinstance(x, group) else [x])
        assert list(got.children) == want
        assert factory(got) == got  # flattening is idempotent


def test_empty_groups_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        pr.All(())
    with pytest.raises(ValueError):
        pr.AnyOf(())


@given(s=_text)
def test_string_literals_survive_both_dialects(s: str) -> None:
    rendered = lance._literal(s)
    assert rendered[0] == rendered[-1] == "'"
    assert rendered[1:-1].replace("''", "'") == s
    assert json.loads(milvus._literal(s)) == s


_LANCE_STRING = re.compile(r"'(?:[^']|'')*'")
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _balanced(text: str) -> bool:
    """Parentheses balance outside string literals (either dialect)."""
    text = _JSON_STRING.sub('""', _LANCE_STRING.sub("''", text))
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


@given(tree=any_tree)
def test_renderers_are_total_and_balanced(tree: pr.Predicate) -> None:
    for render in (lance.render_predicate, milvus.render_predicate):
        out = render(tree)
        assert out and _balanced(out)
    single = pr.All((tree,))
    assert lance.render_predicate(single) == lance.render_predicate(tree)
    assert milvus.render_predicate(single) == milvus.render_predicate(tree)


# ── 4. semantic agreement with a real LanceDB table ───────────────────────

_ROWS = [
    {"name": "alpha", "n": 1, "x": 0.5, "flag": True},
    {"name": "beta", "n": 2, "x": -1.0, "flag": False},
    {"name": "it's", "n": 3, "x": 2.5, "flag": True},
    {"name": "", "n": -4, "x": 0.0, "flag": False},
    {"name": "gamma", "n": 2, "x": 1e3, "flag": True},
]
_COLS = {"name": "str", "n": "int", "x": "float", "flag": "bool"}


def _typed_leaf() -> st.SearchStrategy[pr.Predicate]:
    def leaf_for(
        col: str, values: st.SearchStrategy[Any]
    ) -> st.SearchStrategy[pr.Predicate]:
        f = st.just(col)
        return st.one_of(
            st.builds(pr.eq, f, values),
            st.builds(pr.ne, f, values),
            st.builds(pr.gt, f, values),
            st.builds(pr.lte, f, values),
            st.builds(pr.one_of, f, st.lists(values, min_size=1, max_size=3)),
        )

    return st.one_of(
        leaf_for("name", st.sampled_from(["alpha", "beta", "it's", "", "zeta", "a'b"])),
        leaf_for("n", st.integers(-5, 5)),
        leaf_for("x", st.sampled_from([-1.0, 0.0, 0.5, 2.5, 1e3])),
        leaf_for("flag", st.booleans()),
    )


def _holds(pred: pr.Predicate, row: dict[str, Any]) -> bool:
    if isinstance(pred, pr.Comparison):
        v = row[pred.field]
        ops = {
            "eq": v == pred.value,
            "ne": v != pred.value,
            "gt": v > pred.value,
            "gte": v >= pred.value,
            "lt": v < pred.value,
            "lte": v <= pred.value,
        }
        return ops[pred.operator]
    if isinstance(pred, pr.In):
        return row[pred.field] in pred.values
    if isinstance(pred, pr.All):
        return all(_holds(c, row) for c in pred.children)
    if isinstance(pred, pr.AnyOf):
        return any(_holds(c, row) for c in pred.children)
    raise TypeError(type(pred))


@pytest.fixture(scope="module")
def lance_table(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    import lancedb

    db = lancedb.connect(str(tmp_path_factory.mktemp("lance")))
    return db.create_table("rows", data=_ROWS)


@given(tree=_trees(_typed_leaf()))
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_rendered_filter_selects_the_rows_the_predicate_selects(
    tree: pr.Predicate, lance_table: Any
) -> None:
    where = lance.render_predicate(tree)
    got = sorted(r["n"] for r in lance_table.search().where(where).limit(100).to_list())
    want = sorted(r["n"] for r in _ROWS if _holds(tree, r))
    assert got == want, where


_ = (dt, Path)
