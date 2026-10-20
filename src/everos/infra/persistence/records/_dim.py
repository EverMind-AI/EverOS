"""Vector dimension for the record declarations, read without ``Settings``.

``Vector(dim)`` bakes the dimension into the pydantic core schema when the
record class is created, so the value has to be known at **import** time —
assigning ``_DIM`` later leaves both the Arrow schema and the validator
untouched.

Building the full :class:`~everos.config.Settings` here would make a
malformed ``everos.toml`` crash every CLI command, ``everos config show``
included — the one a user reaches for to fix it. Nothing in this repository
reads settings at import time today, and this module is on the import path of
every subcommand (``cli/main.py`` imports ``commands``, which reaches the
record models). So it reads the single ``[embedding] dim`` key instead, in
the same source order :class:`Settings` uses, and falls back to the
historical default when that cannot be done.

The fallback is deliberately silent and deliberately checked elsewhere:
``infra.persistence.index`` asserts at startup that the dimension baked into
the record classes equals the one ``Settings`` reports, so a disagreement
surfaces as a named error instead of a table quietly built at the wrong
width.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from everos.config import resolve_root

DEFAULT_DIM = 1024
"""Dimension used before ``[embedding] dim`` existed; also the fallback."""

_ENV_VAR = "EVEROS_EMBEDDING__DIM"
_PACKAGED_TOML = Path(__file__).resolve().parents[3] / "config" / "default.toml"


def _dim_from_toml(path: Path) -> object | None:
    """Return the raw ``[embedding] dim`` value, or ``None`` if unreadable.

    Every failure mode collapses to ``None`` on purpose: a missing file, an
    unreadable one and a malformed one all mean "this source has nothing to
    say", and none of them may take the process down (see module docstring).

    ``ValueError`` rather than ``TOMLDecodeError``: a file that is not UTF-8
    fails inside ``tomllib.load``'s own ``decode`` with ``UnicodeDecodeError``,
    which is not a ``TOMLDecodeError`` -- and that would be an import-time
    crash on every CLI command, which is the one thing this module exists to
    avoid. Both are ``ValueError`` subclasses.
    """
    try:
        with path.open("rb") as fh:
            embedding = tomllib.load(fh).get("embedding")
    except (OSError, ValueError):
        return None
    if not isinstance(embedding, dict):
        return None
    return embedding.get("dim")


def configured_dim() -> int:
    """Resolve the vector dimension: env > ``<root>/everos.toml`` > packaged default.

    Mirrors ``Settings``' own source order (``settings.py``: "Source order:
    init_args > env_vars > everos.toml > default.toml"); ``init_args`` has no
    counterpart here because there is no caller to pass them at import time.
    """
    raw: object | None = os.environ.get(_ENV_VAR)
    if raw is None:
        raw = _dim_from_toml(resolve_root() / "everos.toml")
    if raw is None:
        raw = _dim_from_toml(_PACKAGED_TOML)
    try:
        dim = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_DIM
    return dim if dim >= 1 else DEFAULT_DIM
