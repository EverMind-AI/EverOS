from __future__ import annotations

import sys
from pathlib import Path

USE_CASE_ROOT = (
    Path(__file__).resolve().parents[4] / "use-cases" / "cursor-agent-memory"
)
if str(USE_CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(USE_CASE_ROOT))

from hooklib.context import (  # noqa: E402
    count_words,
    format_search_context,
    workspace_recall_query,
)


def test_workspace_recall_query_uses_folder_name() -> None:
    query = workspace_recall_query(["/Users/dev/Projects/EverOS"])
    assert "EverOS" in query


def test_workspace_recall_query_fallback_without_roots() -> None:
    query = workspace_recall_query([])
    assert "project context" in query


def test_count_words() -> None:
    assert count_words("hello world") == 2
    assert count_words("   ") == 0


def test_format_search_context_renders_episodes() -> None:
    data = {
        "episodes": [
            {
                "subject": "Testing preference",
                "episode": "Prefers pytest over unittest.",
                "score": 0.9,
                "atomic_facts": [{"content": "Uses pytest."}],
            }
        ],
        "profiles": [],
    }
    text = format_search_context(data, min_score=0.1)
    assert "EverOS recalled memory" in text
    assert "pytest" in text
    assert "Uses pytest." in text


def test_format_search_context_empty_when_no_hits() -> None:
    assert format_search_context({"episodes": [], "profiles": []}, min_score=0.1) == ""
