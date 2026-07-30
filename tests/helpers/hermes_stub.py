"""Minimal Hermes-runtime stubs for testing the EverOS plugin in isolation.

The EverOS test suite cannot import the real ``agent.memory_provider`` because
Hermes is not a dependency of EverOS. Tests inject this module via
``sys.modules`` so plugin imports resolve to these stubs.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# ── hermes_constants stub ────────────────────────────────────────────────────


def get_hermes_home() -> Path:
    """Return the active Hermes home directory."""
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


# ── tools.registry stub ────────────────────────────────────────────────────────


def tool_error(message: str, **extra: object) -> str:
    """Return a JSON error string matching Hermes's tool_error helper."""
    return json.dumps({"error": str(message), **extra}, ensure_ascii=False)


def tool_result(data: object = None, **kwargs: object) -> str:
    """Return a JSON success string matching Hermes's tool_result helper."""
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


# ── utils stub ─────────────────────────────────────────────────────────────────


def atomic_json_write(
    path: Path,
    data: object,
    *,
    indent: int = 2,
    mode: int | None = None,
    **dump_kwargs: object,
) -> None:
    """Atomic JSON write stripped-down for tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, **dump_kwargs)
    )
    if mode is not None:
        tmp_path.chmod(mode)
    tmp_path.replace(path)


# ── agent.memory_provider stub ───────────────────────────────────────────────


class MemoryProvider(ABC):
    """Minimal ABC matching the methods the EverOS plugin calls/implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider."""

    # Core lifecycle
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs: object) -> None:
        """Initialize for a session."""

    # Optional recall / persistence hooks
    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        return None

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        return None

    def on_turn_start(self, turn_number: int, message: str, **kwargs: object) -> None:
        return None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: object,
    ) -> None:
        return None

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: object,
    ) -> None:
        return None

    # Tools
    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas exposed by the provider."""

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: object
    ) -> str:
        return tool_error(f"Provider does not handle tool {tool_name}")

    # Config
    def get_config_schema(self) -> list[dict[str, Any]]:
        return []

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        return None

    def backup_paths(self) -> list[str]:
        return []

    def shutdown(self) -> None:
        return None
