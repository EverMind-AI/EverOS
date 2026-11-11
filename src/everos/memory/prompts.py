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
"""

from __future__ import annotations

__all__ = ["PromptTemplate"]

import hashlib
from dataclasses import dataclass


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
