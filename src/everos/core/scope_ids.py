"""Validated identifiers used as filesystem path segments.

Sender/owner identifiers retain the historical mixed-case path-safe grammar.
App/project scopes use a stricter portable lowercase grammar because their raw
values become directory names and must not alias across common filesystems.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, StringConstraints

PATH_SAFE_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"
SCOPE_ID_CHARSET = r"^[a-z0-9_.@+-]+$"
SCOPE_ID_MIN_LENGTH = 1
SCOPE_ID_MAX_LENGTH = 128

_PATH_SAFE_RE = re.compile(PATH_SAFE_CHARSET)
_SCOPE_ID_RE = re.compile(SCOPE_ID_CHARSET)
_PATH_TRAVERSAL_TOKENS = frozenset({".", ".."})
_RESERVED_APP_IDS = frozenset(
    {
        ".index",
        ".lock",
        ".projection.lock",
        ".tmp",
        "default_app",
        "everos.toml",
        "ome.toml",
    }
)
_RESERVED_PROJECT_IDS = frozenset({"default_project"})
_WINDOWS_DEVICE_BASENAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def _validate_path_safe(value: str) -> str:
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _PATH_SAFE_RE.match(value):
        raise ValueError(
            "Only alphanumerics, underscore, dot, hyphen, @, and + are allowed"
        )
    return value


def _validate_scope_id(value: str) -> str:
    if not SCOPE_ID_MIN_LENGTH <= len(value) <= SCOPE_ID_MAX_LENGTH:
        raise ValueError("Scope identifiers must contain between 1 and 128 characters")
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _SCOPE_ID_RE.fullmatch(value):
        raise ValueError(
            "Scope identifiers must use lowercase ASCII letters, digits, "
            "underscore, dot, hyphen, @, or +"
        )
    if value.endswith("."):
        raise ValueError("Scope identifiers must not end with a dot")
    if value.split(".", 1)[0] in _WINDOWS_DEVICE_BASENAMES:
        raise ValueError(f"{value!r} is a reserved filesystem device name")
    return value


def validate_app_id(value: str) -> str:
    value = _validate_scope_id(value)
    if value in _RESERVED_APP_IDS:
        raise ValueError(f"{value!r} is a reserved app identifier")
    return value


def validate_project_id(value: str) -> str:
    value = _validate_scope_id(value)
    if value in _RESERVED_PROJECT_IDS:
        raise ValueError(f"{value!r} is a reserved project identifier")
    return value


PathSafeId = Annotated[str, AfterValidator(_validate_path_safe)]
_SCOPE_ID_CONSTRAINTS = StringConstraints(
    min_length=SCOPE_ID_MIN_LENGTH,
    max_length=SCOPE_ID_MAX_LENGTH,
    pattern=SCOPE_ID_CHARSET,
)
AppId = Annotated[str, _SCOPE_ID_CONSTRAINTS, AfterValidator(validate_app_id)]
ProjectId = Annotated[str, _SCOPE_ID_CONSTRAINTS, AfterValidator(validate_project_id)]

__all__ = [
    "PATH_SAFE_CHARSET",
    "SCOPE_ID_CHARSET",
    "SCOPE_ID_MAX_LENGTH",
    "SCOPE_ID_MIN_LENGTH",
    "AppId",
    "PathSafeId",
    "ProjectId",
    "validate_app_id",
    "validate_project_id",
]
