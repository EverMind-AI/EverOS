"""The scanner and the watcher must agree on which md files belong to which kind.

Two matchers exist: the scanner walks the memory root with each kind's glob
(``root.glob(spec.path_glob())``, anchored at the root) and the watcher
classifies one event path with ``KindSpec.matches`` (``PurePosixPath.match``,
which is anchored on the right). The glob docstring states they agree. These
laws generate paths from every registered glob -- the exact shape, then the
same path with components added in front, dropped, or a literal changed --
and check:

1. ``spec.matches(path)`` equals what the scanner's anchored glob would do
   (component count equal and every component matching its pattern);
2. a path matches at most one kind, and ``match_kind`` returns that kind.

The first law failed for paths with extra leading components: the watcher
accepted ``x/app/project/users/u/episodes/episode-2026-01-01.md``, the
scanner never visits it. ``KindSpec.matches`` now requires the component
count to agree.
"""

from __future__ import annotations

import fnmatch
import re

from hypothesis import given
from hypothesis import strategies as st

from everos.memory.cascade.registry import KIND_REGISTRY, KindSpec, match_kind

_segment = st.from_regex(r"[A-Za-z0-9_-]{1,12}", fullmatch=True)


def _instantiate(pattern: str, seg: str) -> str:
    """One concrete component matching one glob component."""
    if pattern == "*":
        return seg
    if "*" in pattern:
        head, _, tail = pattern.partition("*")
        head = head.replace("[0-9]", "7")
        return f"{head}{seg}{tail}"
    return pattern.replace("[0-9]", "7")


def _scanner_would_match(spec: KindSpec, rel: str) -> bool:
    pattern_parts = spec.path_glob().split("/")
    parts = rel.split("/")
    return len(parts) == len(pattern_parts) and all(
        fnmatch.fnmatchcase(p, q) for p, q in zip(parts, pattern_parts, strict=True)
    )


@st.composite
def path_and_spec(draw: st.DrawFn) -> tuple[str, KindSpec]:
    spec = draw(st.sampled_from(KIND_REGISTRY))
    pattern_parts = spec.path_glob().split("/")
    parts = [_instantiate(p, draw(_segment)) for p in pattern_parts]
    variant = draw(st.sampled_from(["exact", "prepend", "drop_first", "mutate"]))
    if variant == "prepend":
        parts = [draw(_segment), *parts]
    elif variant == "drop_first":
        parts = parts[1:]
    elif variant == "mutate":
        i = draw(st.integers(0, len(parts) - 1))
        parts[i] = draw(_segment) + "x"
    return "/".join(parts), spec


@given(case=path_and_spec())
def test_watcher_matcher_agrees_with_the_scanner_glob(
    case: tuple[str, KindSpec],
) -> None:
    rel, spec = case
    assert spec.matches(rel) == _scanner_would_match(spec, rel), rel


@given(case=path_and_spec())
def test_a_path_belongs_to_at_most_one_kind(case: tuple[str, KindSpec]) -> None:
    rel, _ = case
    owners = [s.name for s in KIND_REGISTRY if s.matches(rel)]
    assert len(owners) <= 1, (rel, owners)
    found = match_kind(rel)
    assert (found.name if found else None) == (owners[0] if owners else None)


def test_every_registered_glob_is_absolute_in_shape() -> None:
    for spec in KIND_REGISTRY:
        assert re.fullmatch(r"\*/\*/.+", spec.path_glob()), spec.path_glob()
