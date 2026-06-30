"""Format EverOS search hits for Cursor hook additional_context."""

from __future__ import annotations

from typing import Any


def count_words(text: str) -> int:
    """Rough word count for gating short recall queries."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def workspace_recall_query(workspace_roots: list[str]) -> str:
    """Build a bootstrap recall query from the workspace folder name."""
    if not workspace_roots:
        return "project context and recent decisions"
    name = workspace_roots[0].rstrip("/").split("/")[-1]
    if not name:
        return "project context and recent decisions"
    return f"{name} project context preferences and recent decisions"


def format_search_context(data: dict[str, Any], *, min_score: float) -> str:
    """Turn SearchData into a compact markdown block for the agent."""
    episodes = data.get("episodes") or []
    profiles = data.get("profiles") or []
    if not episodes and not profiles:
        return ""

    lines = [
        "## EverOS recalled memory",
        "",
        "The following memories were retrieved from your local EverOS server.",
        "Treat them as prior context from past Cursor sessions.",
        "",
    ]

    for idx, episode in enumerate(episodes[:5], start=1):
        if not isinstance(episode, dict):
            continue
        score = episode.get("score")
        if isinstance(score, (int, float)) and score < min_score:
            continue
        subject = episode.get("subject") or episode.get("summary") or "Episode"
        body = episode.get("episode") or episode.get("summary") or ""
        lines.append(f"### {idx}. {subject}")
        if body:
            lines.append(str(body).strip())
        facts = episode.get("atomic_facts") or []
        for fact in facts[:3]:
            if isinstance(fact, dict) and fact.get("content"):
                lines.append(f"- {fact['content']}")
        lines.append("")

    for profile in profiles[:1]:
        if not isinstance(profile, dict):
            continue
        pdata = profile.get("profile_data")
        if isinstance(pdata, dict) and pdata:
            lines.append("### User profile")
            for key, value in list(pdata.items())[:8]:
                lines.append(f"- **{key}**: {value}")
            lines.append("")

    return "\n".join(lines).strip()
