"""``summarize_episode``: the model preview replaces the placeholder offline.

Runs the real everalgo ``EpisodeSummarizer`` over the repo's fake LLM. Pins
S2 (the placeholder is recomputed from the event), S3 (compare-and-set per
copy), S4 (the event's language), S5 (a failure writes nothing) and S6 (no
model call once no copy shows the placeholder).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.memory.deps import MemoryDeps
from everos.memory.episode_summary import placeholder_summary
from everos.memory.events import EpisodeCopy, EpisodeExtracted
from everos.memory.models import Episode
from everos.memory.strategies import SUMMARIZE_EPISODE_SPEC, summarize_episode
from tests.fakes import llm_clients, make_deps, user_track_llm, user_track_script
from tests.fakes.stores import InMemoryEpisodeStore

_TEXT = json.loads(user_track_script("episode"))["content"]
_PLACEHOLDER = placeholder_summary(_TEXT)
_MODEL_SUMMARY = json.loads(user_track_script("episode_summary"))["summary"]
_COPIES = [
    EpisodeCopy(owner_id="u_alice", entry_id="ep_20260517_0001"),
    EpisodeCopy(owner_id="u_bob", entry_id="ep_20260517_0004"),
]


def _event(
    *, source: str = "pipeline", language: str | None = None
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_a",
        episode_text=_TEXT,
        episode_timestamp_ms=1_700_000_000_000,
        copies=_COPIES,
        session_id="s1",
        source=source,
        language=language,
    )


def _episode_store(deps: MemoryDeps) -> InMemoryEpisodeStore:
    store = deps.stores.episode
    assert isinstance(store, InMemoryEpisodeStore)
    return store


def _deps(summary_llm: Any, *summaries: str) -> MemoryDeps:
    """Each copy of ``_event()`` in md with the given summary, in copy order."""
    deps = make_deps(llms=llm_clients(summary=summary_llm))
    for copy, summary in zip(_COPIES, summaries, strict=True):
        _episode_store(deps).seed(
            _event().owner_scope(copy),
            copy.entry_id,
            Episode(
                owner_id=copy.owner_id,
                episode=_TEXT,
                summary=summary,
                timestamp=1_700_000_000_000,
                parent_id="mc_a",
            ),
        )
    return deps


def _summaries(deps: MemoryDeps) -> list[str]:
    return [s.item.summary for s in _episode_store(deps).records]


def _requested(llm: FakeLLMClient) -> list[str]:
    return [call.response_format.__name__ for call in llm.calls]


def test_spec() -> None:
    spec = SUMMARIZE_EPISODE_SPEC
    assert spec.name == "summarize_episode"
    assert spec.trigger.on == [EpisodeExtracted]
    assert spec.emits == frozenset()
    assert spec.max_retries == 2


@pytest.mark.parametrize(
    ("source", "applies"), [("pipeline", True), ("reflection", False)]
)
def test_only_pipeline_episodes_are_summarized(source: str, applies: bool) -> None:
    applies_to = SUMMARIZE_EPISODE_SPEC.applies_to
    assert callable(applies_to)
    assert applies_to(_event(source=source), make_deps()) is applies


async def test_one_call_replaces_the_placeholder_of_every_copy() -> None:
    llm = user_track_llm()
    deps = _deps(llm, _PLACEHOLDER, _PLACEHOLDER)

    assert await summarize_episode(_event(), deps) == []

    assert _requested(llm) == ["_SummaryReply"]
    assert _summaries(deps) == [_MODEL_SUMMARY, _MODEL_SUMMARY]


async def test_no_copy_showing_the_placeholder_makes_no_model_call() -> None:
    """S6: an earlier delivery (or an edit) already replaced every copy."""
    llm = user_track_llm()
    deps = _deps(llm, "already written", "edited by hand")

    await summarize_episode(_event(), deps)

    assert llm.calls == []
    assert _summaries(deps) == ["already written", "edited by hand"]


async def test_copies_gone_from_md_make_no_model_call() -> None:
    llm = user_track_llm()
    deps = make_deps(llms=llm_clients(summary=llm))

    await summarize_episode(_event(), deps)

    assert llm.calls == []


async def test_an_edited_summary_is_left_alone() -> None:
    """S3: only copies still showing the placeholder are replaced."""
    llm = user_track_llm()
    deps = _deps(llm, _PLACEHOLDER, "Bob's own words")

    await summarize_episode(_event(), deps)

    assert _summaries(deps) == [_MODEL_SUMMARY, "Bob's own words"]


@pytest.mark.parametrize("language", ["Chinese", "English"])
async def test_the_preview_is_written_in_the_event_language(language: str) -> None:
    """S4: the summarizer is told the episode's language, not left to guess."""
    llm = user_track_llm()
    deps = _deps(llm, _PLACEHOLDER, _PLACEHOLDER)

    await summarize_episode(_event(language=language), deps)

    (call,) = llm.calls
    system = str(call.messages[0].content)
    assert f"Write every output field in {language}" in system
    assert _TEXT in str(call.messages[-1].content)


async def test_a_failed_call_raises_and_writes_nothing() -> None:
    """S5: no fallback cut; the run fails for OME's retries."""

    def _fail(*_args: object, **_kwargs: object) -> str:
        raise TimeoutError("summary endpoint timed out")

    deps = _deps(FakeLLMClient(handler=_fail), _PLACEHOLDER, _PLACEHOLDER)

    with pytest.raises(TimeoutError):
        await summarize_episode(_event(), deps)

    assert _summaries(deps) == [_PLACEHOLDER, _PLACEHOLDER]
