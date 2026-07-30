"""Hermes-agnostic config loading and resolution for the EverOS plugin.

This module deliberately imports only stdlib + the local ``_constants`` /
``_types`` siblings. It must remain importable without a Hermes runtime so
unit tests can exercise it in isolation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ._constants import (
    _DEFAULT_AGENT_ID,
    _DEFAULT_API_URL,
    _DEFAULT_APP_ID,
    _DEFAULT_PROJECT_ID,
    _DEFAULT_USER_ID,
    _SCOPE_ID_CHARSET,
    _SCOPE_ID_MAX_LEN,
    _SCOPE_ID_MIN_LEN,
    _SCOPE_TRAVERSAL_TOKENS,
)
from ._types import ScopeIds

logger = logging.getLogger(__name__)

_SCOPE_ID_RE = re.compile(_SCOPE_ID_CHARSET)
_VALID_MODES = frozenset({"platform", "oss"})


def _atomic_write_text(path: Path, text: str, mode: int | None = None) -> Path:
    """Atomically write ``text`` to ``path`` via a temp file + ``os.replace``.

    If ``mode`` is given the destination is chmod-ed to it (e.g. 0o600).
    Hermes-agnostic — does NOT call ``utils.atomic_json_write``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    if mode is not None:
        # os.replace carries the mode of the temp file, but be explicit in
        # case the destination was pre-created with looser perms.
        os.chmod(path, mode)
    return path


def _defaults() -> dict[str, Any]:
    return {
        "api_url": _DEFAULT_API_URL,
        "mode": "platform",
        "user_id": _DEFAULT_USER_ID,
        "agent_id": _DEFAULT_AGENT_ID,
        "app_id": _DEFAULT_APP_ID,
        "project_id": _DEFAULT_PROJECT_ID,
        "agent_track_enabled": False,
        "everos_root": None,
    }


def load_config(hermes_home: Path) -> dict[str, Any]:
    """Merge defaults with ``$HERMES_HOME/everos.json`` (json wins).

    Env vars ``EVEROS_API_URL`` / ``EVEROS_USER_ID`` / ``EVEROS_AGENT_ID`` /
    ``EVEROS_MODE`` act as a fallback layer between the defaults and the JSON
    file: the JSON file still overrides them when present.

    Returns a plain dict; never raises on a malformed/missing JSON file —
    the defaults are returned instead (mem0 parity).
    """
    cfg = _defaults()

    # Env layer — only set when the operator actually exported a value.
    env_api_url = os.environ.get("EVEROS_API_URL")
    if env_api_url:
        cfg["api_url"] = env_api_url
    env_user_id = os.environ.get("EVEROS_USER_ID")
    if env_user_id:
        cfg["user_id"] = env_user_id
    env_agent_id = os.environ.get("EVEROS_AGENT_ID")
    if env_agent_id:
        cfg["agent_id"] = env_agent_id
    env_mode = os.environ.get("EVEROS_MODE")
    if env_mode:
        cfg["mode"] = env_mode

    config_path = Path(hermes_home) / "everos.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(file_cfg, dict):
                cfg.update(
                    {k: v for k, v in file_cfg.items() if v is not None and v != ""}
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", config_path, exc)

    return cfg


def resolve_user_id(
    config: dict[str, Any],
    kwargs_user_id: str | None,
    kwargs_user_id_alt: str | None,
) -> str:
    """Resolve the canonical user_id.

    Priority: ``config["user_id"]`` (if set and != ``_DEFAULT_USER_ID``) >
    ``kwargs_user_id`` > ``kwargs_user_id_alt`` > ``_DEFAULT_USER_ID``.

    The literal ``_DEFAULT_USER_ID`` is treated as unset (mem0 parity: an
    operator-configured id must be a real id, not the placeholder).
    """
    configured = config.get("user_id")
    if configured and configured != _DEFAULT_USER_ID:
        return configured
    if kwargs_user_id:
        return kwargs_user_id
    if kwargs_user_id_alt:
        return kwargs_user_id_alt
    return _DEFAULT_USER_ID


def resolve_agent_id(config: dict[str, Any]) -> str:
    """Resolve the canonical agent_id."""
    return config.get("agent_id") or _DEFAULT_AGENT_ID


def _validate_scope_id(value: Any, field: str, default: str) -> str:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    if value in _SCOPE_TRAVERSAL_TOKENS:
        raise ValueError(f"{field} must not be a path-traversal token: {value!r}")
    if not _SCOPE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{field} contains characters outside the allowed charset "
            f"{_SCOPE_ID_CHARSET!r}: {value!r}"
        )
    if not (_SCOPE_ID_MIN_LEN <= len(value) <= _SCOPE_ID_MAX_LEN):
        raise ValueError(
            f"{field} length must be in "
            f"[{_SCOPE_ID_MIN_LEN}, {_SCOPE_ID_MAX_LEN}]: {value!r}"
        )
    return value


def get_scope_ids(config: dict[str, Any]) -> ScopeIds:
    """Validate and return the ``(app_id, project_id)`` pair."""
    app_id = _validate_scope_id(config.get("app_id"), "app_id", _DEFAULT_APP_ID)
    project_id = _validate_scope_id(
        config.get("project_id"), "project_id", _DEFAULT_PROJECT_ID
    )
    return ScopeIds(app_id=app_id, project_id=project_id)


def is_configured(config: dict[str, Any]) -> bool:
    """True if ``api_url`` is non-empty or ``mode`` is recognized. No network."""
    api_url = config.get("api_url")
    if api_url:
        return True
    return config.get("mode") in _VALID_MODES
