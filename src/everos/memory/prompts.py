"""Registered prompt templates: the prompt text EverOS itself owns.

Algorithm prompts belong to everalgo. A few live here (docs/group-memory.md,
section 11): the group roster block, the group-profile update and the
constraint-binding call. Each is a :class:`PromptTemplate` with an id, a
version and a fingerprint of its text, pinned by tests, so data built from one
version can be told apart from data built from another, and the id survives a
move to everalgo.

Invariant: a change to any part's text changes the fingerprint; a change that
is meant to ship bumps the version in the same commit, and the test that pins
both is updated with it.

A reply model of one of these prompts may give fields a parsing default (a
key the model leaves out reads as empty); :data:`LENIENT_REPLY` keeps the
JSON schema the call sends as strict as one without defaults.
"""

from __future__ import annotations

__all__ = ["LENIENT_REPLY", "PromptTemplate"]

import hashlib
from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict


@dataclass(frozen=True)
class PromptTemplate:
    """A registered prompt: id, version and the text parts it renders.

    ``fingerprint`` changes whenever a part's text does, so data built from
    one version can be told apart from data built from another.
    """

    id: str
    version: int
    parts: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256("\n".join(self.parts).encode("utf-8")).hexdigest()[:16]


def _without_defaults(schema: dict[str, Any]) -> None:
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)


LENIENT_REPLY = ConfigDict(json_schema_extra=_without_defaults)
"""``model_config`` for a reply model whose fields have parsing defaults: the
schema sent still requires every field and names no default, byte for byte
the schema of the same model without them."""
