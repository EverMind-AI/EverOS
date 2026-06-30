#!/usr/bin/env python3
"""Cursor sessionEnd hook — flush buffered messages into EverOS memory."""

from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401  # pyright: ignore[reportMissingImports]
from hooklib.config import EverOSHookConfig
from hooklib.everos_client import EverOSError, flush, health


def main() -> None:
    raw = sys.stdin.read()
    try:
        hook_input = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        hook_input = {}

    cfg = EverOSHookConfig.load()
    if not cfg.is_configured():
        sys.exit(0)

    conversation_id = hook_input.get("conversation_id")
    if not conversation_id:
        sys.exit(0)

    if not health(cfg.base_url):
        if cfg.debug:
            print("EverOS health check failed", file=sys.stderr)
        sys.exit(0)

    session_id = cfg.session_id_for(str(conversation_id))
    try:
        flush(
            base_url=cfg.base_url,
            session_id=session_id,
            app_id=cfg.app_id,
            project_id=cfg.project_id,
        )
    except EverOSError as exc:
        if cfg.debug:
            print(f"EverOS flush failed: {exc}", file=sys.stderr)
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
