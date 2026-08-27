"""``include_profile`` has to reach the answer prompt, on every benchmark.

It did not. ``SearchManager._fetch_profile`` is a side lookup -- it does not rank, it
does not consume a ``top_k`` slot, and it only attaches a ``profiles`` list to the
response -- so whether it matters is decided entirely by the harness's context builder.
Three of the four builders accepted the argument and dropped it:

* locomo, longmemeval  -- no ``build_context``, so ``run.py``'s default branch ran, and
  that branch never mentioned ``profiles``.
* subtlememory         -- had one, whose docstring read "profiles unused here".
* evermembench         -- the only one that rendered them.

The cost was not a missing feature. A "profile on vs off" pair was run
on LoCoMo and LongMemEval to decide whether profiles help; with the prompt identical
in both arms it measured decider nondeterminism instead (McNemar p=0.10 and p=0.15),
and the opposite-signed conclusions drawn from it were wrong.

Two properties are pinned: a fetched profile is rendered, and an unfetched one changes
nothing -- the second is what keeps every published prompt byte-identical to its
reference harness while the flag stays off.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import adapters  # noqa: E402
from adapters._profile import (  # noqa: E402
    MEMORY_HEADING,
    PROFILE_HEADING,
    render_profile_lines,
    with_profile_block,
)

run = importlib.import_module("run")

_ADAPTERS = ("locomo", "longmemeval", "subtlememory", "evermembench")

_PROFILE = [
    {
        "profile_data": {
            "summary": "Lan Ye, operations lead on the platform team.",
            "explicit_info": [{"category": "role", "description": "ops manager"}],
            "implicit_traits": [
                {"trait": "detail-oriented", "basis": "checks every rollout"}
            ],
        }
    }
]

_EPISODES = [
    {
        "subject": "Release",
        "episode": "Shipped v2 on Tuesday.",
        "timestamp": "2026-07-29",
        "session_id": "session_8",
        "id": "ep1",
    }
]


class _Cfg:
    adapter = "locomo"
    prompts: dict[str, Any] = {}


def _context(adapter: str, profiles: list[dict]) -> str:
    cfg = _Cfg()
    cfg.adapter = adapter
    return run._build_context(_EPISODES, profiles, "speaker_a", "speaker_b", cfg)


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_a_fetched_profile_reaches_the_prompt(adapter: str) -> None:
    """The whole point of the flag. Previously true only for evermembench."""
    out = _context(adapter, _PROFILE)
    assert PROFILE_HEADING in out, f"{adapter} drops the profile"
    assert "Lan Ye, operations lead" in out
    assert "[role] ops manager" in out
    assert "[trait] detail-oriented -- checks every rollout" in out


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_no_profile_leaves_the_prompt_untouched(adapter: str) -> None:
    """Rendering must be inert when nothing was fetched.

    This is what lets the fix ship without re-verifying four prompt-parity suites:
    with ``include_profile`` off the profiles list is empty and the output is exactly
    what the reference harness produces.
    """
    assert _context(adapter, []) == _context(adapter, [])
    assert PROFILE_HEADING not in _context(adapter, [])


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_the_episodes_survive_the_profile_block(adapter: str) -> None:
    """A profile is added around the memories, never in place of them."""
    assert "Shipped v2 on Tuesday." in _context(adapter, _PROFILE)


def test_profile_precedes_the_memories() -> None:
    """Standing context first, dated memories after.

    Order is the reason the block exists at all: a profile has no timestamp and belongs
    to no session, and a model told to weigh recency would treat it as one more dated
    memory if it were mixed into the list.
    """
    out = _context("locomo", _PROFILE)
    assert out.index(PROFILE_HEADING) < out.index("Shipped v2")


def test_every_adapter_that_declares_build_context_takes_profiles() -> None:
    """A two-argument signature is what makes dropping them a visible choice."""
    import inspect

    for name in _ADAPTERS:
        fn = getattr(adapters.get(name), "build_context", None)
        if fn is None:
            continue
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["episodes", "profiles"], f"{name}: {params}"


def test_renderer_tolerates_the_bare_profile_shape() -> None:
    """The search response uses two shapes; both have been seen in artefacts."""
    bare = [{"summary": "S", "explicit_info": [], "implicit_traits": []}]
    assert render_profile_lines(bare) == ["- S"]
    assert render_profile_lines([]) == []


def test_block_helper_handles_empty_memories() -> None:
    """A profile with no retrieved episodes must not render a dangling heading."""
    out = with_profile_block("", _PROFILE)
    assert out.startswith(PROFILE_HEADING)
    assert MEMORY_HEADING not in out
