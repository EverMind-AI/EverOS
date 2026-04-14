"""
OpenClaw adapter for EverMemOS evaluation pipeline.

Wraps OpenClaw memory lifecycle (ingest / flush / index / search / get) via a
Node bridge, exposes a BaseAdapter-compatible surface, and emits
session-level retrieval traces and lifecycle diagnostics alongside the
shared answer prompt.

This file currently contains only the registration stub; the real add() /
search() / answer() machinery is built out in subsequent tasks (Task 4-8).
"""
from typing import Any, List

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("openclaw")
class OpenClawAdapter(BaseAdapter):
    def __init__(self, config: dict, output_dir: Any = None):
        super().__init__(config)
        self.output_dir = output_dir

    async def add(self, conversations: List[Conversation], **kwargs) -> Any:
        raise NotImplementedError("add() is implemented in Task 4")

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        raise NotImplementedError("search() is implemented in Task 5")

    def get_system_info(self) -> dict:
        return {"name": "OpenClaw", "config": self.config}
