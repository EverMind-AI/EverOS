"""Validated identifiers used as app/project filesystem scope segments."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator

PATH_SAFE_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"

_PATH_SAFE_RE = re.compile(PATH_SAFE_CHARSET)
_PATH_TRAVERSAL_TOKENS = frozenset({".", ".."})
_RESERVED_APP_IDS = frozenset({"default_app"})
_RESERVED_PROJECT_IDS = frozenset({"default_project"})


def _validate_path_safe(value: str) -> str:
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _PATH_SAFE_RE.match(value):
        raise ValueError(
            "Only alphanumerics, underscore, dot, hyphen, @, and + are allowed"
        )
    return value


def validate_app_id(value: str) -> str:
    value = _validate_path_safe(value)
    if value in _RESERVED_APP_IDS:
        raise ValueError("'default_app' is a reserved app identifier")
    return value


def validate_project_id(value: str) -> str:
    value = _validate_path_safe(value)
    if value in _RESERVED_PROJECT_IDS:
        raise ValueError("'default_project' is a reserved project identifier")
    return value


PathSafeId = Annotated[str, AfterValidator(_validate_path_safe)]
AppId = Annotated[str, AfterValidator(validate_app_id)]
ProjectId = Annotated[str, AfterValidator(validate_project_id)]

__all__ = [
    "PATH_SAFE_CHARSET",
    "AppId",
    "PathSafeId",
    "ProjectId",
    "validate_app_id",
    "validate_project_id",
]
