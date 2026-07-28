"""Markdown-to-index sync daemon (cascade).

Watcher (realtime fs events) + scanner (periodic walk) + worker
(claim/drain) keep LanceDB in sync with the md files under the memory
root. Cascade is the *only* path that writes LanceDB; service / entry
points just write md and trust the daemon to catch up.

Public surface — what lifespan providers / CLI commands import:

- :class:`CascadeOrchestrator` — composite owner; start / stop / sync.
- :class:`CascadeConfig` — construction-time tuning knobs.
- :data:`KIND_REGISTRY` / :func:`match_kind` — kind dispatch (also
  used by CLI ``cascade sync --path`` to resolve a single file's kind).
"""

from .orchestrator import CascadeConfig as CascadeConfig
from .orchestrator import CascadeOrchestrator as CascadeOrchestrator
from .registry import KIND_REGISTRY as KIND_REGISTRY
from .registry import KindSpec as KindSpec
from .registry import match_kind as match_kind

__all__ = [
    "KIND_REGISTRY",
    "CascadeConfig",
    "CascadeOrchestrator",
    "KindSpec",
    "match_kind",
]
