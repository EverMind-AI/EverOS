"""Hermes-side CLI for the EverOS memory plugin (``hermes everos <subcommand>``).

Runs in the Hermes process when EverOS is the active memory provider. The
module is loaded *before* the provider is initialised (Hermes' plugin CLI
discovery imports ``cli.py`` purely to build the argparse tree), so it stays
lightweight: stdlib + the local ``._constants`` / ``._config`` helpers only.
``httpx`` and ``hermes_constants`` are lazy-imported inside command handlers
so a partially-installed environment fails clearly at call time instead of
at import time.

Subcommands:

- ``status``   — reachability + active mode/user/scope + breaker state.
- ``search``   — one-off search against the active config; prints JSON.
- ``flush``    — POST ``/memory/flush`` for a session.
- ``setup``    — non-interactive shortcut that writes/merges ``everos.json``.

Config lives at ``$HERMES_HOME/everos.json`` (see ``integrations/hermes/
README.md``). Secret ``EVEROS_API_KEY`` belongs in ``~/.hermes/.env``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from ._config import _atomic_write_text
from ._constants import (
    _DEFAULT_AGENT_ID,
    _DEFAULT_API_URL,
    _DEFAULT_APP_ID,
    _DEFAULT_PROJECT_ID,
    _DEFAULT_SEARCH_METHOD,
    _DEFAULT_TOOL_SEARCH_TOP_K,
    _DEFAULT_USER_ID,
)

logger = logging.getLogger(__name__)

_CONFIG_FILE_NAME = "everos.json"
_OWNER_CHOICES = ("user", "agent")

# Substrings that mark a config key as secret: anything matching these is
# redacted when echoing config to stdout so secrets never leak via `setup`.
_SECRET_KEY_SUFFIXES = ("_key", "_token")


def _redact(values: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``values`` with secret fields masked.

    Any key ending in ``_key`` or ``_token`` (e.g. ``api_key``,
    ``refresh_token``) is replaced with ``"***"`` so ``hermes everos setup``
    can safely echo the written values without leaking secrets to stdout.
    """
    return {
        k: ("***" if isinstance(k, str) and k.endswith(_SECRET_KEY_SUFFIXES) else v)
        for k, v in values.items()
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    """Resolve ``$HERMES_HOME`` (env or ``~/.hermes``).

    Prefers ``hermes_constants.get_hermes_home()`` when available so the
    plugin honours Hermes' own resolution logic.
    """
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

        return Path(get_hermes_home())
    except Exception:
        env = os.environ.get("HERMES_HOME")
        if env:
            return Path(env).expanduser()
        return Path.home() / ".hermes"


def _default_config() -> dict[str, Any]:
    return {
        "mode": "platform",
        "api_url": _DEFAULT_API_URL,
        "api_key": os.environ.get("EVEROS_API_KEY", ""),
        "user_id": _DEFAULT_USER_ID,
        "agent_id": _DEFAULT_AGENT_ID,
        "app_id": _DEFAULT_APP_ID,
        "project_id": _DEFAULT_PROJECT_ID,
    }


def _load_config() -> dict[str, Any]:
    """Load config from defaults + ``$HERMES_HOME/everos.json`` overrides."""
    cfg = _default_config()
    path = _hermes_home() / _CONFIG_FILE_NAME
    if path.is_file():
        try:
            file_cfg = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(file_cfg, dict):
                cfg.update({k: v for k, v in file_cfg.items() if v not in (None, "")})
        except Exception as exc:  # noqa: BLE001 - best-effort config read
            logger.warning("Failed to read %s: %s", path, exc)
    return cfg


def _save_config(values: dict[str, Any]) -> Path:
    """Merge ``values`` into ``$HERMES_HOME/everos.json`` (atomic-ish)."""
    home = _hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / _CONFIG_FILE_NAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:  # noqa: BLE001 - best-effort
            existing = {}
    existing.update({k: v for k, v in values.items() if v is not None})
    text = json.dumps(existing, indent=2, sort_keys=True)
    _atomic_write_text(path, text, mode=0o600)
    return path


# ---------------------------------------------------------------------------
# EverOS HTTP client helpers
# ---------------------------------------------------------------------------


def _client(cfg: dict[str, Any]):
    """Return a configured httpx client bound to the active ``api_url``."""
    import httpx  # type: ignore[import-not-found]

    headers: dict[str, str] = {}
    api_key = cfg.get("api_key") or os.environ.get("EVEROS_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(
        base_url=str(cfg.get("api_url", _DEFAULT_API_URL)).rstrip("/"),
        headers=headers,
        timeout=10.0,
    )


def _health_check(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lightweight GET ``/health`` probe."""
    try:
        resp = _client(cfg).get("/health", timeout=5.0)
        return {
            "reachable": True,
            "status_code": resp.status_code,
            "body": resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text,
        }
    except Exception as exc:  # noqa: BLE001 - network probe
        return {"reachable": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes everos`` argparse subcommand tree.

    Called by Hermes' plugin CLI registration system during argparse setup.
    The *subparser* is the parser for ``hermes everos``.
    """
    subs = subparser.add_subparsers(dest="everos_action")

    subs.add_parser("status", help="Show EverOS reachability and active config.")

    search_p = subs.add_parser(
        "search", help="One-off memory search; prints JSON results."
    )
    search_p.add_argument("query", help="Search query.")
    search_p.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOOL_SEARCH_TOP_K,
        help=f"Max results (default: {_DEFAULT_TOOL_SEARCH_TOP_K}).",
    )
    search_p.add_argument(
        "--method",
        default=_DEFAULT_SEARCH_METHOD,
        choices=["keyword", "vector", "hybrid", "agentic"],
        help=f"Search method (default: {_DEFAULT_SEARCH_METHOD}).",
    )
    search_p.add_argument(
        "--owner",
        choices=list(_OWNER_CHOICES),
        default=_OWNER_CHOICES[0],
        help="Scope owner: 'user' (user_id) or 'agent' (agent_id). "
        "Default: user. Independent of --mode.",
    )

    flush_p = subs.add_parser(
        "flush", help="Flush the current session's buffered messages."
    )
    flush_p.add_argument(
        "--session-id",
        default="",
        help="Session id to flush (defaults to a CLI-derived session).",
    )
    flush_p.add_argument(
        "--owner",
        choices=list(_OWNER_CHOICES),
        default=_OWNER_CHOICES[0],
        help="Scope owner: 'user' (user_id) or 'agent' (agent_id). "
        "Default: user. Independent of --mode.",
    )

    setup_p = subs.add_parser(
        "setup",
        help="Non-interactive shortcut: write/merge everos.json.",
    )
    setup_p.add_argument("--mode", choices=["platform", "oss"], default="platform")
    setup_p.add_argument("--api-url", default="", help="EverOS API base URL.")
    setup_p.add_argument("--api-key", default="", help="EverOS API key.")
    setup_p.add_argument("--user-id", default="", help="User identifier.")
    setup_p.add_argument("--agent-id", default="", help="Agent identifier.")
    setup_p.add_argument("--app-id", default="", help="EverOS app_id (scope).")
    setup_p.add_argument("--project-id", default="", help="EverOS project_id (scope).")

    subparser.set_defaults(func=everos_command)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def everos_command(args: argparse.Namespace) -> int:
    action = getattr(args, "everos_action", None)
    if not action:
        print("Usage: hermes everos {status|search|flush|setup}")
        return 2
    try:
        if action == "status":
            _cmd_status(args)
        elif action == "search":
            _cmd_search(args)
        elif action == "flush":
            _cmd_flush(args)
        elif action == "setup":
            _cmd_setup(args)
        else:
            print(f"Unknown everos action: {action}")
            return 2
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("everos command '%s' failed: %s", action, exc)
        print(f"everos {action}: {exc}")
        return 1


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _active_scope(
    cfg: dict[str, Any], *, owner: str = _OWNER_CHOICES[0]
) -> dict[str, Any]:
    """Return the user_id/agent_id/scope block for the given owner.

    ``owner`` selects ``user_id`` vs ``agent_id`` directly, independent of
    ``mode`` (which is only ever ``platform`` or ``oss``).
    """
    mode = cfg.get("mode", "platform")
    scope: dict[str, Any] = {
        "mode": mode,
        "app_id": cfg.get("app_id", _DEFAULT_APP_ID),
        "project_id": cfg.get("project_id", _DEFAULT_PROJECT_ID),
    }
    if owner == _OWNER_CHOICES[1]:
        scope["agent_id"] = cfg.get("agent_id", _DEFAULT_AGENT_ID)
    else:
        scope["user_id"] = cfg.get("user_id", _DEFAULT_USER_ID)
    return scope


def _breaker_state() -> dict[str, Any] | None:
    """Best-effort read of the active provider's circuit-breaker state.

    The provider runs in the Hermes process; if it is loaded and exposes
    breaker state, surface it. Otherwise return ``None``.
    """
    try:
        from agent import memory_provider  # type: ignore[import-not-found]

        provider = getattr(memory_provider, "get_active_provider", lambda: None)()
        if provider is None:
            return None
        breaker = {}
        for attr in ("_consecutive_failures", "_breaker_open_until"):
            val = getattr(provider, attr, None)
            if val is not None:
                breaker[attr.lstrip("_")] = val
        return breaker or None
    except Exception:  # noqa: BLE001 - best-effort
        return None


def _cmd_status(args: argparse.Namespace) -> None:
    cfg = _load_config()
    health = _health_check(cfg)
    scope = _active_scope(cfg)
    breaker = _breaker_state()
    print(
        json.dumps(
            {
                "config": {
                    "api_url": cfg.get("api_url"),
                    "mode": cfg.get("mode"),
                },
                "scope": scope,
                "health": health,
                "circuit_breaker": breaker,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_search(args: argparse.Namespace) -> None:
    cfg = _load_config()
    scope = _active_scope(cfg, owner=args.owner)
    body: dict[str, Any] = {
        "query": args.query,
        "method": args.method,
        "top_k": args.top_k,
        "app_id": scope["app_id"],
        "project_id": scope["project_id"],
    }
    if args.owner == _OWNER_CHOICES[1]:
        body["agent_id"] = scope["agent_id"]
    else:
        body["user_id"] = scope["user_id"]
    resp = _client(cfg).post("/api/v1/memory/search", json=body)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2, sort_keys=True))


def _cmd_flush(args: argparse.Namespace) -> None:
    cfg = _load_config()
    scope = _active_scope(cfg, owner=args.owner)
    fallback = scope.get("agent_id") or scope.get("user_id")
    session_id = args.session_id or f"everos-cli-{fallback}"
    body: dict[str, Any] = {
        "session_id": session_id,
        "app_id": scope["app_id"],
        "project_id": scope["project_id"],
    }
    resp = _client(cfg).post("/api/v1/memory/flush", json=body)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2, sort_keys=True))


def _cmd_setup(args: argparse.Namespace) -> None:
    values: dict[str, Any] = {"mode": args.mode}
    if args.api_url:
        values["api_url"] = args.api_url
    if args.api_key:
        values["api_key"] = args.api_key
    if args.user_id:
        values["user_id"] = args.user_id
    if args.agent_id:
        values["agent_id"] = args.agent_id
    if args.app_id:
        values["app_id"] = args.app_id
    if args.project_id:
        values["project_id"] = args.project_id
    path = _save_config(values)
    print(f"Wrote {path}")
    print(json.dumps(_redact(values), indent=2, sort_keys=True))
