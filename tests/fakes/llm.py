"""Scripted replies for everalgo operators, shaped as their strict reply models.

A :class:`~everalgo.testing.FakeLLMClient` handler receives the
``response_format`` each operator requests, so a test answers per operator
by the reply model's name instead of by prompt text.

Canned replies live in ``llm_scripts/<track>/*.json``, one file per reply
(the JSON text the model returns), for the ``user_track``, the
``agent_track`` and ``knowledge`` (document topic extraction). Each
user / agent operator has a default script and an empty (or blank) one for
its skip / failure path::

    llm = user_track_llm()                         # every default script
    llm = user_track_llm(boundary="boundary_empty")  # one operator overridden
    llm = agent_track_llm(case_filter="case_filter_empty")
"""

from __future__ import annotations

__all__ = [
    "AGENT_TRACK_OPERATORS",
    "KNOWLEDGE_OPERATORS",
    "UNUSABLE_REPLY",
    "USER_TRACK_OPERATORS",
    "agent_track_llm",
    "agent_track_script",
    "boundary_reply",
    "boundary_window",
    "knowledge_llm",
    "knowledge_script",
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

_SCRIPTS = Path(__file__).parent / "llm_scripts"
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

AGENT_TRACK_OPERATORS: Mapping[str, str] = {
    "boundary": "_BatchReply",
    "case_filter": "_CaseFilterReply",
    "case_compress": "_CaseCompressReply",
    "case_gate": "_SuccessProbability",
    "skill_cluster": "_ClusterAssignmentReply",
    "skill_extract": "_SkillExtractReply",
}
"""Operator name -> reply model, as :data:`USER_TRACK_OPERATORS`, for the agent
track."""


KNOWLEDGE_OPERATORS: Mapping[str, str] = {
    "topic_tree": "TopicTreeReply",
    "category": "CategoryReply",
    "topic_split": "TopicSplitReply",
    "uncovered_assign": "UncoveredAssignReply",
}
"""Operator name -> reply model for document topic extraction. The default
``topic_tree`` script is one topic over the first block of a one-paragraph
document, so the split / uncovered passes are not reached on it."""


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
    tail_complete: bool = False,
    **scripts: str,
) -> FakeLLMClient:
    """A model for the memorize path, answering by the reply model requested.

    Boundary and episode replies are scripted here; agent-track operators
    answer from their default agent-track scripts.

    Args:
        boundaries: Either a queue of cut lists, one popped per boundary call
            (``[]`` once empty), or a function of the window size returning
            the cuts.
        episode_title: Every episode reply's ``title``.
        episode_content: Every episode reply's ``content`` (and ``summary``).
        calls: When given, the requested reply model's name is appended per
            call.
        tail_complete: Every boundary reply's verdict on the uncut tail.
        **scripts: ``operator=script`` overrides for the other user-track
            operators, as for :func:`user_track_llm`.

    Raises:
        KeyError: An override names an unknown user-track operator.

    Other user-track operators, and knowledge extraction, answer from their
    canned scripts; anything else gets an unusable reply (its background
    strategy logs and gives up).
    """
    queue = None if callable(boundaries) else [list(b) for b in boundaries]
    episode = json.dumps(
        {"content": episode_content, "summary": episode_content, "title": episode_title}
    )
    canned = {
        **_canned("agent_track", AGENT_TRACK_OPERATORS, {}),
        **_canned("knowledge", KNOWLEDGE_OPERATORS, {}),
        **_user(scripts),
    }

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
        if name == "_EpisodeReply":
            return episode
        return canned.get(name, UNUSABLE_REPLY)

    return FakeLLMClient(handler=_answer)


def user_track_script(name: str) -> str:
    """The canned reply ``llm_scripts/user_track/<name>.json``."""
    return _script("user_track", name)


def agent_track_script(name: str) -> str:
    """The canned reply ``llm_scripts/agent_track/<name>.json``."""
    return _script("agent_track", name)


def knowledge_script(name: str) -> str:
    """The canned reply ``llm_scripts/knowledge/<name>.json``."""
    return _script("knowledge", name)


def knowledge_llm(**scripts: str) -> FakeLLMClient:
    """A model answering every knowledge-extraction call from its canned script.

    Args:
        **scripts: ``operator=script`` overrides, as for :func:`user_track_llm`.

    Raises:
        KeyError: An override names an unknown operator.
    """
    return _scripted_llm(
        "knowledge", _canned("knowledge", KNOWLEDGE_OPERATORS, scripts)
    )


def user_track_llm(**scripts: str) -> FakeLLMClient:
    """A model answering every user-track operator from its canned script.

    Args:
        **scripts: ``operator=script`` overrides, e.g. ``episode="episode_blank"``.

    Raises:
        KeyError: An override names an unknown operator.
    """
    return _scripted_llm("user-track", _user(scripts))


def agent_track_llm(**scripts: str) -> FakeLLMClient:
    """A model answering every agent-track operator from its canned script.

    Args:
        **scripts: ``operator=script`` overrides, e.g.
            ``case_filter="case_filter_empty"``.

    Raises:
        KeyError: An override names an unknown operator.
    """
    return _scripted_llm(
        "agent-track", _canned("agent_track", AGENT_TRACK_OPERATORS, scripts)
    )


def _scripted_llm(track: str, by_reply: Mapping[str, str]) -> FakeLLMClient:
    def _answer(_messages: Any, *, response_format: type, **_kwargs: Any) -> str:
        try:
            return by_reply[response_format.__name__]
        except KeyError:
            raise AssertionError(
                f"no {track} script answers {response_format.__name__}"
            ) from None

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
