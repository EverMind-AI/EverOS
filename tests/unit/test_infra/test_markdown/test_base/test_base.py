"""Tests for Frontmatter base classes (chassis layer)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.infra.persistence.markdown import (
    BaseFrontmatter,
    GroupScopedFrontmatter,
    UserScopedFrontmatter,
)


def test_base_requires_id_and_type() -> None:
    with pytest.raises(ValidationError):
        BaseFrontmatter()  # type: ignore[call-arg]


def test_base_default_schema_version_is_one() -> None:
    fm = BaseFrontmatter(id="x", type="t")
    assert fm.schema_version == 1


def test_base_extra_fields_allowed() -> None:
    """L2 / L3 / L4 fields ride along without subclass declaration."""
    fm = BaseFrontmatter(
        id="x",
        type="t",
        md_sha256="abc",  # L2
        last_indexed_at="2026-04-22T10:00:00Z",
        custom_user_field="anything",  # L4
    )
    dumped = fm.model_dump()
    assert dumped["md_sha256"] == "abc"
    assert dumped["custom_user_field"] == "anything"


def test_user_scoped_track_default() -> None:
    fm = UserScopedFrontmatter(id="x", type="t", user_id="u_jason")
    assert fm.track == "user"


def test_user_scoped_requires_user_id() -> None:
    with pytest.raises(ValidationError):
        UserScopedFrontmatter(id="x", type="t")  # type: ignore[call-arg]


def test_group_scoped_track_default() -> None:
    fm = GroupScopedFrontmatter(id="x", type="t", group_id="trip")
    assert fm.track == "group"


def test_group_scoped_requires_group_id() -> None:
    with pytest.raises(ValidationError):
        GroupScopedFrontmatter(id="x", type="t")  # type: ignore[call-arg]


def test_track_literal_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        UserScopedFrontmatter(id="x", type="t", user_id="u", track="agent")


def test_scope_dir_classvars() -> None:
    """Scope mixins declare the top-level memory-root subdirectory."""
    assert BaseFrontmatter.SCOPE_DIR == ""  # scope-agnostic by default
    assert UserScopedFrontmatter.SCOPE_DIR == "users"
    assert GroupScopedFrontmatter.SCOPE_DIR == "groups"
