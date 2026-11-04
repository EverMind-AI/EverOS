"""C11: every daily-log write takes ``only_if``, keyword-only, with no default.

A writer that forgets its source check must fail loudly (``TypeError`` at the
call), never write unchecked: a skipped check is how a deleted memory comes
back.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from everos.infra.persistence.markdown import (
    MarkdownAgentCaseStore,
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
    MarkdownForesightStore,
)
from everos.memory.ports import (
    AgentCaseStore,
    AtomicFactStore,
    EpisodeStore,
    ForesightStore,
)
from tests.fakes.stores import (
    InMemoryAgentCaseStore,
    InMemoryAtomicFactStore,
    InMemoryEpisodeStore,
    InMemoryForesightStore,
)

_WRITES: list[tuple[type, str]] = [
    (EpisodeStore, "append"),
    (EpisodeStore, "append_many"),
    (EpisodeStore, "deprecate"),
    (AtomicFactStore, "append_many"),
    (AtomicFactStore, "deprecate"),
    (ForesightStore, "append_many"),
    (AgentCaseStore, "append"),
]
_ADAPTERS = {
    EpisodeStore: (MarkdownEpisodeStore, InMemoryEpisodeStore),
    AtomicFactStore: (MarkdownAtomicFactStore, InMemoryAtomicFactStore),
    ForesightStore: (MarkdownForesightStore, InMemoryForesightStore),
    AgentCaseStore: (MarkdownAgentCaseStore, InMemoryAgentCaseStore),
}


def _cases() -> list[Any]:
    cases = []
    for port, name in _WRITES:
        for owner in (port, *_ADAPTERS[port]):
            cases.append(
                pytest.param(getattr(owner, name), id=f"{owner.__name__}.{name}")
            )
    return cases


@pytest.mark.parametrize("method", _cases())
def test_only_if_is_keyword_only_without_default(method: Callable[..., Any]) -> None:
    p = inspect.signature(method).parameters["only_if"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty
