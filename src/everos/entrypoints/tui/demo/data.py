"""Scripted story data for the educational ``everos demo`` flow."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MEMORY_SEED = "I love climbing in Yosemite every spring."
DEFAULT_QUERY = "Where do I like to climb?"


@dataclass(frozen=True, slots=True)
class DemoStory:
    """Small, deterministic story rendered by the demo TUI."""

    owner: str
    memory: str
    query: str
    answer: str
    source_filename: str
    fact_filename: str


def default_demo_story() -> DemoStory:
    """Return the cinematic story used by README media and no-prompt previews."""

    return DemoStory(
        owner="alice",
        memory=DEFAULT_MEMORY_SEED,
        query="Where does Alice like to climb?",
        answer="Yosemite every spring",
        source_filename="episode-2026-06-20.md",
        fact_filename="atomic_fact-2026-06-20.md",
    )


def build_demo_story(
    memory_seed: str | None = None,
    query: str | None = None,
) -> DemoStory:
    """Build a playable demo story from one user memory and one recall query."""

    memory = _clean(memory_seed, DEFAULT_MEMORY_SEED)
    recall_query = _clean(query, DEFAULT_QUERY)
    return DemoStory(
        owner="you",
        memory=memory,
        query=recall_query,
        answer=_derive_demo_answer(memory),
        source_filename="episode-demo.md",
        fact_filename="atomic_fact-demo.md",
    )


def _clean(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    stripped = value.strip()
    return stripped or fallback


def _derive_demo_answer(memory: str) -> str:
    """Keep the demo deterministic without pretending to run the server."""

    lower_memory = memory.lower()
    if "yosemite" in lower_memory:
        if "spring" in lower_memory:
            return "Yosemite every spring"
        return "Yosemite"
    return _compact(memory)


def _compact(text: str, *, limit: int = 66) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
