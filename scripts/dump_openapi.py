"""Dump the FastAPI OpenAPI schema to ``docs/openapi.json``.

Static export — does **not** start the server. Calls ``app.openapi()``
directly on the FastAPI instance returned by ``create_app()``, which
the runtime ``GET /openapi.json`` handler returns verbatim. No lifespan
is run, so this is fast and side-effect-free.

Modes:

* default — write ``docs/openapi.json``.
* ``--check`` — write to a temp file and ``diff`` against the on-disk
  copy. Exits non-zero on drift, so it can be wired into ``make lint``
  to fail PRs that touch the API surface without regenerating the
  committed schema. Same shape as ``check_datetime_discipline.py``.

Run::

    python scripts/dump_openapi.py            # write docs/openapi.json
    python scripts/dump_openapi.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _ROOT / "docs" / "openapi.json"


def _build_schema() -> dict:
    """Return the FastAPI app's full OpenAPI schema.

    Force ``ENV=DEV`` so the ``openapi_url`` route is enabled — without
    it the dev-mode endpoint check (see ``app.py``) shadows the route.
    The schema content itself is identical in dev vs prod; the flag only
    controls whether the runtime ``GET /openapi.json`` is exposed. We
    flip it here so the static export matches the dev-mode endpoint
    output the e2e test compares against.
    """
    import os

    os.environ["ENV"] = "DEV"
    # Local import so an import-time evaluation of ``ENV`` (read inside
    # ``create_app``) sees the override above.
    from everos.entrypoints.api.app import create_app

    # Pass an empty lifespan list so we don't pull up SQLite / LanceDB /
    # OME — the schema is computed from static route declarations alone.
    app = create_app(lifespan_providers=[])
    return app.openapi()


def _render(schema: dict) -> str:
    """Pretty-print the schema as JSON with stable key order + trailing newline."""
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _write_target(content: str) -> None:
    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_text(content, encoding="utf-8")


def _check_against_target(content: str) -> int:
    if not _TARGET.is_file():
        print(
            f"error: {_TARGET.relative_to(_ROOT)} does not exist; "
            f"run `make openapi` to generate it.",
            file=sys.stderr,
        )
        return 1
    existing = _TARGET.read_text(encoding="utf-8")
    if existing == content:
        print(f"OK — {_TARGET.relative_to(_ROOT)} matches app.openapi() output.")
        return 0
    # Drift: print a unified diff to stderr so CI / reviewer can see what changed.
    import difflib

    diff = "".join(
        difflib.unified_diff(
            existing.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"{_TARGET.relative_to(_ROOT)} (committed)",
            tofile="app.openapi() (current)",
        )
    )
    # Limit to first ~200 lines so a giant schema rewrite stays scannable.
    capped = "".join(diff.splitlines(keepends=True)[:200])
    print(
        f"error: {_TARGET.relative_to(_ROOT)} is out of date.\n"
        "Run `make openapi` and commit the result.\n\n" + capped,
        file=sys.stderr,
    )
    if len(diff.splitlines()) > 200:
        print(
            f"... (truncated; full diff is {len(diff.splitlines())} lines)",
            file=sys.stderr,
        )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against docs/openapi.json without writing; exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    schema = _build_schema()
    content = _render(schema)

    if args.check:
        return _check_against_target(content)

    _write_target(content)
    print(f"wrote {_TARGET.relative_to(_ROOT)} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    # Silence the unused-import warning on tempfile (kept for future use).
    _ = tempfile
    sys.exit(main())
