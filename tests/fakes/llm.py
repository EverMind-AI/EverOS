"""Scripted replies for everalgo operators, shaped as their strict reply models.

A :class:`~everalgo.testing.FakeLLMClient` handler receives the
``response_format`` each operator requests, so a test answers per operator
by the reply model's name instead of by prompt text.

The user track's canned replies live in ``llm_scripts/user_track/*.json``,
one file per reply (the JSON text the model returns). Each operator has a
default script and an empty (or blank) one for its skip / failure path::

    llm = user_track_llm()                         # every default script
    llm = user_track_llm(boundary="boundary_empty")  # one operator overridden
"""

from __future__ import annotations

__all__ = [
    "UNUSABLE_REPLY",
    "USER_TRACK_OPERATORS",
    "boundary_reply",
    "boundary_window",
    "memorize_llm",
    "user_track_llm",
    "user_track_script",
]

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient

UNUSABLE_REPLY = "this is not JSON"
"""A reply no operator can parse: it raises ``LLMOutputError``."""

_USER_TRACK = Path(__file__).parent / "llm_scripts" / "user_track"
_WINDOW = re.compile(r'<messages count="(\d+)">')

USER_TRACK_OPERATORS: Mapping[str, str] = {
    "boundary": "_BatchReply",
    "episode": "_EpisodeReply",
    "atomic_facts": "_AtomicFactsReply",
    "profile_init": "_ProfileItems",
    "profile_update": "_MemCellUpdateOutput",
    "reflect": "_ReflectOutput",
}
"""Operator name -> the reply model it requests; the default script shares the name."""


def boundary_reply(
    boundaries: Sequence[int] = (), *, tail_complete: bool = False
) -> str:
    """A batch boundary reply cutting after each 1-based message index given."""
    return json.dumps(
        {
            "reasoning": "scripted",
            "boundaries": list(boundaries),
            "tail_complete": tail_complete,
        }
    )


def boundary_window(messages: Sequence[Any]) -> int:
    """How many chat messages a boundary request shows the model."""
    found = _WINDOW.search(messages[-1].content)
    if found is None:
        raise AssertionError("not a batch boundary request")
    return int(found.group(1))


def memorize_llm(
    boundaries: Sequence[Sequence[int]] | Callable[[int], Sequence[int]] = (),
    *,
    episode_title: str = "Test Subject",
    episode_content: str = "Test body",
    calls: list[str] | None = None,
    **scripts: str,
) -> FakeLLMClient:
    """A model for the memorize path, answering by the reply model requested.

    Args:
        boundaries: Either a queue of cut lists, one popped per boundary call
            (``[]`` once empty), or a function of the window size returning
            the cuts.
        episode_title: Every episode reply's ``title``.
        episode_content: Every episode reply's ``content`` (and ``summary``).
        calls: When given, the requested reply model's name is appended per
            call.
        **scripts: ``operator=script`` overrides for the other user-track
            operators, as for :func:`user_track_llm`.

    Other user-track operators answer from their canned scripts; anything
    else gets an unusable reply (its background strategy logs and gives up).
    """
    queue = None if callable(boundaries) else [list(b) for b in boundaries]
    episode = json.dumps(
        {"content": episode_content, "summary": episode_content, "title": episode_title}
    )
    canned = _canned(scripts)

    def _answer(messages: Any, *, response_format: type, **_kwargs: Any) -> str:
        name = response_format.__name__
        if calls is not None:
            calls.append(name)
        if name == "_BatchReply":
            if queue is None:
                assert callable(boundaries)
                return boundary_reply(boundaries(boundary_window(messages)))
            return boundary_reply(queue.pop(0) if queue else [])
        if name == "_EpisodeReply":
            return episode
        return canned.get(name, UNUSABLE_REPLY)

    return FakeLLMClient(handler=_answer)


def user_track_script(name: str) -> str:
    """The canned reply ``llm_scripts/user_track/<name>.json``."""
    return (_USER_TRACK / f"{name}.json").read_text(encoding="utf-8")


def user_track_llm(**scripts: str) -> FakeLLMClient:
    """A model answering every user-track operator from its canned script.

    Args:
        **scripts: ``operator=script`` overrides, e.g. ``episode="episode_blank"``.

    Raises:
        KeyError: An override names an unknown operator.
    """
    by_reply = _canned(scripts)

    def _answer(_messages: Any, *, response_format: type, **_kwargs: Any) -> str:
        try:
            return by_reply[response_format.__name__]
        except KeyError:
            raise AssertionError(
                f"no user-track script answers {response_format.__name__}"
            ) from None

    return FakeLLMClient(handler=_answer)


def _canned(scripts: Mapping[str, str]) -> dict[str, str]:
    """Reply model name -> canned reply, with ``operator=script`` overrides.

    Raises:
        KeyError: An override names an unknown operator.
    """
    unknown = set(scripts) - set(USER_TRACK_OPERATORS)
    if unknown:
        raise KeyError(f"unknown user-track operators: {sorted(unknown)}")
    return {
        reply: user_track_script(scripts.get(operator, operator))
        for operator, reply in USER_TRACK_OPERATORS.items()
    }
