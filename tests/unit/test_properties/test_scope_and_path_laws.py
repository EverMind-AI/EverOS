"""Laws of the id, scope-directory and path-segment helpers.

These functions sit between an identifier a client sends and the directory
it lands in, so a law that fails here is either a path escape or two scopes
silently sharing a directory.

1. ``app_id_from_dir(app_dir_name(x)) == x`` and the ``project_*`` pair
   likewise, for every id the API accepts in that field. Before this test
   the API also accepted ``app_id="default_app"`` and
   ``project_id="default_project"``: those are the directory names of the
   default scope, so such an app shared the default app's directory and its
   index rows were read back as ``default``. Each field now rejects its own
   aliasing name and nothing else (a ``sender_id`` never goes through the
   mapping).
2. An accepted id joined under a root stays inside the root.
3. ``sanitize_dirname`` is idempotent, returns exactly one path component,
   never ``""`` / ``.`` / ``..`` and never more than 50 characters; the
   skill-name sanitizer is idempotent and its directory segment is always
   ``skill_`` plus a traversal-free component.
4. ``content_sha256`` is deterministic and does not depend on key order.
   It is not claimed injective, and it is not: the canonical form joins
   ``key=value`` lines, so ``{"a": "1\\nb=2"}`` hashes like
   ``{"a": "1", "b": "2"}``. Handler values are section bodies, which can
   contain newlines, so a collision needs a value that reproduces another
   key's line exactly -- recorded, not fixed, because changing the
   canonical form would re-index every entry once.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from everos.core.persistence import memory_root as mr
from everos.core.persistence.markdown.path_safety import sanitize_dirname
from everos.entrypoints.api.routes.memorize import (
    _reject_path_traversal,
    _reject_reserved_scope_dir,
)
from everos.infra.persistence.markdown.mds.agent_skill import AgentSkillFrontmatter
from everos.memory.cascade.handlers._common import content_sha256

_candidate_ids = st.text("abcXYZ019_.@+-", min_size=1, max_size=20)


_check_app = _reject_reserved_scope_dir("default_app")
_check_project = _reject_reserved_scope_dir("default_project")


def _accepted(check: Callable[[str], str]) -> Callable[[str], bool]:
    def ok(value: str) -> bool:
        try:
            check(value)
        except ValueError:
            return False
        return True

    return ok


accepted_app_ids = _candidate_ids.filter(_accepted(_check_app))
accepted_project_ids = _candidate_ids.filter(_accepted(_check_project))
accepted_ids = _candidate_ids.filter(_accepted(_reject_path_traversal))
free_text = st.text(max_size=80)


@given(x=accepted_app_ids, y=accepted_project_ids)
def test_scope_dir_names_invert_for_every_accepted_id(x: str, y: str) -> None:
    assert mr.app_id_from_dir(mr.app_dir_name(x)) == x
    assert mr.project_id_from_dir(mr.project_dir_name(y)) == y


def test_only_the_aliasing_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="directory name of the default scope"):
        _check_app("default_app")
    with pytest.raises(ValueError, match="directory name of the default scope"):
        _check_project("default_project")
    assert _check_app("default_project") == "default_project"
    assert _check_project("default_app") == "default_app"
    assert _reject_path_traversal("default_app") == "default_app"  # sender ids


@given(x=accepted_app_ids, y=accepted_project_ids)
def test_accepted_ids_stay_under_the_root(x: str, y: str) -> None:
    # Resolution needs no real directory; resolve the root too, or on Windows
    # the drive letter the target gains makes it "not relative" to the root.
    root = Path("/memory-root").resolve()
    target = (root / mr.app_dir_name(x) / mr.project_dir_name(y)).resolve()
    assert target.is_relative_to(root)
    assert len(target.relative_to(root).parts) == 2


@given(raw=free_text, fallback=st.sampled_from(["fallback", "x"]))
def test_sanitize_dirname_is_idempotent_and_one_safe_component(
    raw: str, fallback: str
) -> None:
    out = sanitize_dirname(raw, fallback)
    assert sanitize_dirname(out, fallback) == out
    assert out not in ("", ".", "..")
    assert "/" not in out and "\\" not in out and "\x00" not in out
    assert len(out) <= 50
    assert len(Path(out).parts) == 1


@given(raw=free_text)
def test_skill_name_sanitizer_is_idempotent_and_its_dir_is_safe(raw: str) -> None:
    fm = AgentSkillFrontmatter
    name = fm.sanitize_skill_name(raw)
    assert fm.sanitize_skill_name(name) == name
    segment = fm.skill_dir_name(raw)
    assert segment == fm.skill_dir_name(name)
    assert segment.startswith(fm.SKILL_DIR_PREFIX)
    assert "/" not in segment and "\\" not in segment
    assert segment not in (".", "..")


@given(
    parts=st.dictionaries(
        st.text("abcdef_", min_size=1, max_size=8), st.text(max_size=40), max_size=6
    ),
    seed=st.integers(0, 1000),
)
def test_content_sha256_is_deterministic_and_order_independent(
    parts: dict[str, str], seed: int
) -> None:
    import random

    keys = list(parts)
    random.Random(seed).shuffle(keys)
    shuffled = {k: parts[k] for k in keys}
    assert content_sha256(parts) == content_sha256(shuffled)
    assert content_sha256(parts) == content_sha256(dict(parts))
