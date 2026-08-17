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
    score: float = 0.0


def default_demo_story() -> DemoStory:
    """Return the cinematic story used by README media and no-prompt previews.

    This is only the static showcase content for ``--plain`` / ``--cinematic``.
    The interactive demo builds its story from real server recall (see
    :func:`everos.entrypoints.tui.demo.cloud.search_recall`).
    """

    return DemoStory(
        owner="alice",
        memory=DEFAULT_MEMORY_SEED,
        query="Where does Alice like to climb?",
        answer="Yosemite every spring",
        source_filename="episode-2026-06-20.md",
        fact_filename="atomic_fact-2026-06-20.md",
    )
