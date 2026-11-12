"""Registered prompt templates: the prompt text EverOS itself owns.

Algorithm prompts belong to everalgo. A few live here (docs/group-memory.md,
section 11): the chat episode prompt, the group roster block, the
group-profile update and the constraint-binding call. Each is a
:class:`PromptTemplate` with an id, a version and a fingerprint of its text,
pinned by tests, so data built from one version can be told apart from data
built from another, and the id survives a move to everalgo.

The everalgo prompts EverOS sends as everalgo wrote them are recorded too,
each as an :class:`AlgoPrompt`: the package it comes from and the fingerprint
of its text, pinned by a test against the vendored build. The manifest of
both (:mod:`everos.memory.prompt_manifest`) is what a store's builder records.

Invariant: a change to any part's text changes the fingerprint; a change that
is meant to ship bumps the version in the same commit, and the test that pins
both is updated with it.

A reply model of one of these prompts may give fields a parsing default (a
key the model leaves out reads as empty); :data:`LENIENT_REPLY` keeps the
JSON schema the call sends as strict as one without defaults.
"""

from __future__ import annotations

__all__ = ["LENIENT_REPLY", "AlgoPrompt", "PromptTemplate"]

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict


def _fingerprint(parts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


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
        return _fingerprint(self.parts)


@dataclass(frozen=True)
class AlgoPrompt:
    """An everalgo prompt EverOS sends unchanged: where it comes from, its text.

    Attributes:
        id: ``everalgo.<package>.<prompt>``, stable across everalgo versions.
        package: The everalgo distribution that ships it
            (``everalgo-user-memory``).
        parts: Its templates as the installed package defines them, the
            placeholders unfilled.
    """

    id: str
    package: str
    parts: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.parts)


def _without_defaults(schema: dict[str, Any]) -> None:
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)


LENIENT_REPLY = ConfigDict(json_schema_extra=_without_defaults)
"""``model_config`` for a reply model whose fields have parsing defaults: the
schema sent still requires every field and names no default, byte for byte
the schema of the same model without them."""
