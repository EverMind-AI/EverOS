"""Parse Cursor / Claude-style JSONL composer transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnContent:
    user: str
    assistant: str

    @property
    def has_content(self) -> bool:
        return bool(self.user.strip() or self.assistant.strip())

    @property
    def fingerprint(self) -> str:
        return f"{self.user}\n---\n{self.assistant}"


def _text_from_user_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        return "\n\n".join(parts)
    return ""


def _text_from_assistant_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        return "\n\n".join(parts)
    return ""


def extract_last_turn(lines: list[str]) -> TurnContent:
    """Extract the latest user/assistant text from a JSONL transcript.

    Compatible with Claude Code / Cursor composer transcripts that mark
    turn boundaries with ``{"type":"system","subtype":"turn_duration"}``.
    When the stop hook runs, the current turn's duration marker may not
    exist yet, so the active turn spans from the last marker to EOF.
    """
    turn_start = 0
    for index in range(len(lines) - 1, -1, -1):
        try:
            entry = json.loads(lines[index])
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "system" and entry.get("subtype") == "turn_duration":
            turn_start = index + 1
            break

    user_parts: list[str] = []
    assistant_parts: list[str] = []

    for line in lines[turn_start:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message") or {}
        content = message.get("content")
        if entry.get("type") == "user":
            text = _text_from_user_content(content)
            if text.strip():
                user_parts.append(text.strip())
        elif entry.get("type") == "assistant":
            text = _text_from_assistant_content(content)
            if text.strip():
                assistant_parts.append(text.strip())

    return TurnContent(
        user="\n\n".join(user_parts),
        assistant="\n\n".join(assistant_parts),
    )


def read_transcript_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]
