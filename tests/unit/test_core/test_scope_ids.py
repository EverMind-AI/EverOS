"""Portable filesystem contracts for app and project scope identifiers."""

from __future__ import annotations

import pytest

from everos.core.scope_ids import validate_app_id, validate_project_id


@pytest.mark.parametrize(
    "value",
    [
        "default",
        "app",
        "project-1",
        "team_one",
        "release.2",
        "user@example.com",
        "tag+value",
        "x" * 128,
    ],
)
def test_scope_ids_accept_portable_lowercase_values(value: str) -> None:
    assert validate_app_id(value) == value
    assert validate_project_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "Project-A",
        "safe\n",
        "contains space",
        "contains/slash",
        "trailing.",
        "x" * 129,
    ],
)
def test_scope_ids_reject_nonportable_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_app_id(value)
    with pytest.raises(ValueError):
        validate_project_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "con",
        "con.txt",
        "prn",
        "aux.log",
        "nul",
        "com1",
        "com9.json",
        "lpt1",
        "lpt9.txt",
    ],
)
def test_scope_ids_reject_windows_device_names(value: str) -> None:
    with pytest.raises(ValueError, match="filesystem device"):
        validate_app_id(value)
    with pytest.raises(ValueError, match="filesystem device"):
        validate_project_id(value)


@pytest.mark.parametrize(
    "value",
    [
        ".index",
        ".lock",
        ".projection.lock",
        ".tmp",
        "default_app",
        "everos.toml",
        "ome.toml",
    ],
)
def test_app_ids_reject_memory_root_managed_names(value: str) -> None:
    with pytest.raises(ValueError, match="reserved app"):
        validate_app_id(value)


def test_project_id_rejects_default_directory_alias() -> None:
    with pytest.raises(ValueError, match="reserved project"):
        validate_project_id("default_project")
