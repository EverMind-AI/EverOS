#!/usr/bin/env python3
"""Cursor stop hook — append the latest turn to EverOS via /add."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # pyright: ignore[reportMissingImports]
from hooklib.config import EverOSHookConfig
from hooklib.everos_client import EverOSError, add_messages, health, message_item
from hooklib.transcript import extract_last_turn, read_transcript_lines


def _state_path(root: Path, conversation_id: str) -> Path:
    state_dir = root / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(conversation_id.encode()).hexdigest()[:16]
    return state_dir / f"{digest}.json"


def _already_saved(state_file: Path, fingerprint: str) -> bool:
    if not state_file.is_file():
        return False
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return state.get("fingerprint") == fingerprint


def _mark_saved(state_file: Path, fingerprint: str) -> None:
    state_file.write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    raw = sys.stdin.read()
    try:
        hook_input = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        hook_input = {}

    cfg = EverOSHookConfig.load()
    if not cfg.is_configured():
        sys.exit(0)

    transcript_path = hook_input.get("transcript_path")
    conversation_id = hook_input.get("conversation_id")
    if not transcript_path or not conversation_id:
        sys.exit(0)

    if not health(cfg.base_url):
        if cfg.debug:
            print("EverOS health check failed", file=sys.stderr)
        sys.exit(0)

    try:
        lines = read_transcript_lines(str(transcript_path))
    except OSError as exc:
        if cfg.debug:
            print(f"transcript read failed: {exc}", file=sys.stderr)
        sys.exit(0)

    turn = extract_last_turn(lines)
    if not turn.has_content:
        sys.exit(0)

    hook_root = Path(__file__).resolve().parent
    state_file = _state_path(hook_root, str(conversation_id))
    fingerprint = turn.fingerprint
    if _already_saved(state_file, fingerprint):
        sys.exit(0)

    session_id = cfg.session_id_for(str(conversation_id))
    messages: list[dict] = []
    if turn.user.strip():
        messages.append(
            message_item(
                sender_id=cfg.user_id,
                role="user",
                content=turn.user.strip(),
            )
        )
    if turn.assistant.strip():
        messages.append(
            message_item(
                sender_id=cfg.user_id,
                role="assistant",
                content=turn.assistant.strip(),
            )
        )
    if not messages:
        sys.exit(0)

    try:
        add_messages(
            base_url=cfg.base_url,
            session_id=session_id,
            app_id=cfg.app_id,
            project_id=cfg.project_id,
            messages=messages,
        )
    except EverOSError as exc:
        if cfg.debug:
            print(f"EverOS add failed: {exc}", file=sys.stderr)
        sys.exit(0)

    _mark_saved(state_file, fingerprint)
    sys.exit(0)


if __name__ == "__main__":
    main()
