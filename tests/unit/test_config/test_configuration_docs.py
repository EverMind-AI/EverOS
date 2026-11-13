"""Every configuration key is documented in docs/configuration.md.

Each section's table there lists its keys (``| `key` | ...``) under the
section's heading (``### `[section]` ``). A key added to :class:`Settings`
without a row fails here, and so does a row for a key that no longer exists.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from everos.config import Settings

_DOC = Path(__file__).resolve().parents[3] / "docs" / "configuration.md"
_HEADING = re.compile(r"^#{3,4} `\[([a-z_.]+)\]`")
_ROW = re.compile(r"^\| `([a-z_0-9]+)` \|")


def _keys(model: type[BaseModel], prefix: tuple[str, ...] = ()) -> Iterator[str]:
    for name, field in model.model_fields.items():
        sub = field.annotation
        if isinstance(sub, type) and issubclass(sub, BaseModel):
            yield from _keys(sub, (*prefix, name))
        else:
            yield f"[{'.'.join(prefix)}] {name}"


def _documented() -> set[str]:
    found: set[str] = set()
    section: str | None = None
    for line in _DOC.read_text(encoding="utf-8").splitlines():
        heading = _HEADING.match(line)
        if heading:
            section = heading.group(1)
        elif line.startswith("#"):
            section = None
        elif section is not None and (row := _ROW.match(line)):
            found.add(f"[{section}] {row.group(1)}")
    return found


def test_every_key_has_a_row_and_every_row_a_key() -> None:
    keys = set(_keys(Settings))
    documented = _documented()
    assert sorted(keys - documented) == [], "undocumented keys"
    assert sorted(documented - keys) == [], "rows for keys that do not exist"
