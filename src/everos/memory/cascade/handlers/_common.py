"""Helpers shared by every daily-log handler.

The inline-block parsers undo
:func:`everos.core.persistence.markdown.entries._render_value` —
the writer stringifies values with ``str()`` (and list / tuple as
``[a, b, c]``), so callers that need a typed value back must reverse
the rendering here. Each helper raises ``ValueError`` on malformed
input so the worker classifies the failure as unrecoverable (no point
retrying a YAML / format error).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import PurePosixPath

from everos.component.utils.datetime import from_iso_format
from everos.core.persistence import app_id_from_dir, project_id_from_dir


def content_sha256(named_parts: dict[str, str]) -> str:
    """Canonical SHA-256 over a ``{key: value}`` mapping.

    Each handler declares its ``content_change_keys`` (a tuple of
    ``"section:Name"`` / ``"inline:name"`` / handler-specific
    identifiers) and collects the corresponding values from the parsed
    md. This helper canonicalises that mapping (sorted key, joined
    ``key=value`` per line) and hashes — so two runs over the same
    content always produce the same digest regardless of declaration
    order.

    Audit / scope fields (owner_id / session_id / timestamp / etc.)
    are intentionally NOT included by any handler — changes to them
    don't propagate to LanceDB and don't waste an embedding call.
    """
    canonical = "\n".join(f"{k}={named_parts[k]}" for k in sorted(named_parts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_owner(frontmatter: dict[str, object], md_path: str) -> tuple[str, str]:
    """Pull ``owner_id`` / ``owner_type`` from the frontmatter + path.

    Layout is ``<app>/<project>/<scope>/<owner>/...``, so ``owner_type`` is
    encoded by the **scope segment** (path index 2: ``users`` → ``"user"``,
    ``agents`` → ``"agent"``) — NOT the path prefix, which now leads with
    ``<app>/<project>``. ``owner_id`` rides on the user_id / agent_id
    frontmatter field.
    """
    parts = PurePosixPath(md_path).parts
    scope_dir = parts[2] if len(parts) > 2 else ""
    if scope_dir == "agents":
        owner_type = "agent"
        owner_id = str(frontmatter.get("agent_id", ""))
    else:
        owner_type = "user"
        owner_id = str(frontmatter.get("user_id", ""))
    return owner_id, owner_type


def resolve_scope(md_path: str) -> tuple[str, str]:
    """Recover ``(app_id, project_id)`` from an md path's leading two segments.

    Inverse of the writer's layout ``<app>/<project>/<scope>/...``: the
    reserved dir names ``default_app`` / ``default_project`` map back to the
    ``"default"`` id (see :func:`everos.core.persistence.app_id_from_dir`).
    A path missing the prefix degrades to the default space rather than
    raising — cascade should never see one, since every writer emits the
    prefix, but a defensive fallback keeps a malformed path from crashing
    the worker.
    """
    parts = PurePosixPath(md_path).parts
    app_dir = parts[0] if len(parts) > 0 else "default_app"
    project_dir = parts[1] if len(parts) > 1 else "default_project"
    return app_id_from_dir(app_dir), project_id_from_dir(project_dir)


def require_iso_timestamp(raw: str | None, *, field: str = "timestamp") -> _dt.datetime:
    """Parse a required ISO-8601 timestamp from an inline value."""
    if not raw:
        raise ValueError(f"entry inline is missing required {field!r}")
    return from_iso_format(raw.strip())


def optional_iso_timestamp(raw: str | None) -> _dt.datetime | None:
    """Parse an optional ISO-8601 timestamp; ``None`` / empty → ``None``."""
    if not raw or not raw.strip():
        return None
    return from_iso_format(raw.strip())


def optional_int(raw: str | None) -> int | None:
    """Parse an optional int from an inline value (``"7"`` → ``7``)."""
    if not raw or not raw.strip():
        return None
    return int(raw.strip())


def require_float(raw: str | None, *, field: str) -> float:
    """Parse a required float inline value."""
    if not raw or not raw.strip():
        raise ValueError(f"entry inline is missing required {field!r}")
    return float(raw.strip())


def parse_inline_list(raw: str) -> list[str]:
    """Parse ``"[a, b, c]"`` back into ``["a", "b", "c"]``.

    Mirrors :func:`everos.core.persistence.markdown.entries._render_value`
    output shape (list / tuple inline values). Empty / malformed
    payloads yield an empty list — the writer never emits literal
    commas inside ids, so simple split-on-comma is safe.
    """
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    body = text[1:-1].strip()
    if not body:
        return []
    return [tok.strip() for tok in body.split(",") if tok.strip()]
