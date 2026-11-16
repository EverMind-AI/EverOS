"""Scripted replies for everalgo operators, shaped as their strict reply models.

A :class:`~everalgo.testing.FakeLLMClient` handler receives the
``response_format`` each operator requests, so a test answers per operator
by the reply model's name instead of by prompt text.

Canned replies live in ``llm_scripts/user_track/*.json``, one file per reply
(the JSON text the model returns). Each operator has a default script and an
empty (or blank) one for its skip / failure path::

    llm = user_track_llm()                         # every default script
    llm = user_track_llm(boundary="boundary_empty")  # one operator overridden
"""

from __future__ import annotations

__all__ = [
    "DECIDER_KEEP_FIRST_TEN",
    "UNUSABLE_REPLY",
    "USER_TRACK_OPERATORS",
    "boundary_reply",
    "boundary_window",
    "memorize_llm",
    "prompt_text",
    "quote_every_message",
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

DECIDER_KEEP_FIRST_TEN = json.dumps({"core": list(range(10)), "next_queries": []})
"""A multi-round decider reply: keep the first ten candidates, stop."""

_SCRIPTS = Path(__file__).parent / "llm_scripts"
_WINDOW = re.compile(r'<messages count="(\d+)">|</messages>\n<count>(\d+)</count>')
"""The window size: everalgo's rendering, or EverOS's with the count after the
records (``memory.extract.boundary.prompt_order``)."""

_BLOCK = re.compile(
    r'^ {20}"speaker": (.*?),\n {20}"content": (.*?)\n {16}\}$',
    re.MULTILINE | re.DOTALL,
)
"""A message of the original episode request (one pseudo-JSON block)."""

_CUED = {
    "CuedEpisodeReply": "EpisodeReply",
    "CuedImageEpisodeReply": "ImageEpisodeReply",
}
"""The episode reply models a call with the profile-cue block asks for, and the
model of the same call without it."""


def prompt_text(messages: Sequence[Any]) -> str:
    """Every text part of ``messages``, joined."""
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content or []:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def quote_every_message(
    prompt: str, *, keep: Callable[[str], bool] | None = None
) -> list[dict[str, Any]]:
    """Profile cues quoting each message of an episode request (each one
    ``keep`` accepts): the speaker as rendered and the message's start."""
    return [
        {"speaker": speaker.strip(), "quote": content.strip()[:40]}
        for speaker, content in _BLOCK.findall(prompt)
        if content.strip() and (keep is None or keep(content))
    ]


def _with_cues(reply: str, messages: Sequence[Any]) -> str:
    """``reply`` with profile cues quoting every message of the request."""
    data = json.loads(reply)
    data["profile_cues"] = quote_every_message(prompt_text(messages))
    return json.dumps(data)


USER_TRACK_OPERATORS: Mapping[str, str] = {
    "boundary": "_BatchReply",
    "episode": "EpisodeReply",
    "profile_update": "_ProfileUpdateReply",
    "profile_consolidate": "_ConsolidateReply",
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
    return int(found.group(1) or found.group(2))


def memorize_llm(
    boundaries: Sequence[Sequence[int]] | Callable[[int], Sequence[int]] = (),
    *,
    episode_title: str = "Test Subject",
    episode_content: str = "Test body",
    calls: list[str] | None = None,
    tail_complete: bool = False,
    **scripts: str,
) -> FakeLLMClient:
    """A model for the memorize path, answering by the reply model requested.

    Boundary and episode replies are scripted here.

    Args:
        boundaries: Either a queue of cut lists, one popped per boundary call
            (``[]`` once empty), or a function of the window size returning
            the cuts.
        episode_title: Every episode reply's ``title``.
        episode_content: Every episode reply's ``content``.
        calls: When given, the requested reply model's name is appended per
            call.
        tail_complete: Every boundary reply's verdict on the uncut tail.
        **scripts: ``operator=script`` overrides for the other user-track
            operators, as for :func:`user_track_llm`.

    Raises:
        KeyError: An override names an unknown user-track operator.

    Other user-track operators answer from their canned scripts; the
    multi-round search's decider keeps the first ten candidates and stops
    (:data:`DECIDER_KEEP_FIRST_TEN`); anything else gets an unusable reply (its
    background strategy logs and gives up).
    """
    queue = None if callable(boundaries) else [list(b) for b in boundaries]
    episode = json.dumps({"title": episode_title, "content": episode_content})
    canned = _user(scripts)

    def _answer(messages: Any, *, response_format: type, **_kwargs: Any) -> str:
        name = response_format.__name__
        if calls is not None:
            calls.append(name)
        if name == "_BatchReply":
            if queue is None:
                assert callable(boundaries)
                cuts = boundaries(boundary_window(messages))
            else:
                cuts = queue.pop(0) if queue else []
            return boundary_reply(cuts, tail_complete=tail_complete)
        if name == "EpisodeReply":
            return episode
        if name == "CuedEpisodeReply":
            # The profile-cue block: every message is marked.
            return _with_cues(episode, messages)
        if name == "_DecisionReply":
            return DECIDER_KEEP_FIRST_TEN
        return canned.get(name, UNUSABLE_REPLY)

    return FakeLLMClient(handler=_answer)


def user_track_script(name: str) -> str:
    """The canned reply ``llm_scripts/user_track/<name>.json``."""
    return _script("user_track", name)


def user_track_llm(**scripts: str) -> FakeLLMClient:
    """A model answering every user-track operator from its canned script.

    Args:
        **scripts: ``operator=script`` overrides, e.g. ``episode="episode_blank"``.

    Raises:
        KeyError: An override names an unknown operator.
    """
    return _scripted_llm("user-track", _user(scripts))


def _scripted_llm(track: str, by_reply: Mapping[str, str]) -> FakeLLMClient:
    def _answer(messages: Any, *, response_format: type, **_kwargs: Any) -> str:
        name = response_format.__name__
        if name in _CUED and _CUED[name] in by_reply:
            # The profile-cue block: the script's episode, every message marked.
            return _with_cues(by_reply[_CUED[name]], messages)
        try:
            return by_reply[name]
        except KeyError:
            raise AssertionError(f"no {track} script answers {name}") from None

    return FakeLLMClient(handler=_answer)


def _script(track: str, name: str) -> str:
    return (_SCRIPTS / track / f"{name}.json").read_text(encoding="utf-8")


def _user(scripts: Mapping[str, str]) -> dict[str, str]:
    return _canned("user_track", USER_TRACK_OPERATORS, scripts)


def _canned(
    track: str, operators: Mapping[str, str], scripts: Mapping[str, str]
) -> dict[str, str]:
    """Reply model name -> canned reply, with ``operator=script`` overrides.

    Raises:
        KeyError: An override names an unknown operator.
    """
    unknown = set(scripts) - set(operators)
    if unknown:
        raise KeyError(f"unknown {track} operators: {sorted(unknown)}")
    return {
        reply: _script(track, scripts.get(operator, operator))
        for operator, reply in operators.items()
    }
