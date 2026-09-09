#!/usr/bin/env python3
"""Cursor sessionStart hook — recall EverOS memories into additional_context."""

from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401  # pyright: ignore[reportMissingImports]
from hooklib.config import EverOSHookConfig
from hooklib.context import format_search_context, workspace_recall_query
from hooklib.everos_client import EverOSError, health, search


def main() -> None:
    raw = sys.stdin.read()
    try:
        hook_input = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        hook_input = {}

    cfg = EverOSHookConfig.load()
    if not cfg.is_configured():
        sys.exit(0)

    if not health(cfg.base_url):
        if cfg.debug:
            print("EverOS health check failed", file=sys.stderr)
        sys.exit(0)

    roots = hook_input.get("workspace_roots") or []
    query = workspace_recall_query(roots if isinstance(roots, list) else [])

    try:
        data = search(
            base_url=cfg.base_url,
            user_id=cfg.user_id,
            query=query,
            app_id=cfg.app_id,
            project_id=cfg.project_id,
            top_k=cfg.top_k,
            min_score=cfg.min_score,
        )
    except EverOSError as exc:
        if cfg.debug:
            print(f"EverOS search failed: {exc}", file=sys.stderr)
        sys.exit(0)

    context = format_search_context(data, min_score=cfg.min_score)
    if not context:
        sys.exit(0)

    print(json.dumps({"additional_context": context}))


if __name__ == "__main__":
    main()
