"""Markdown-to-index sync daemon (cascade).

Watcher (realtime fs events) + scanner (periodic walk) + worker
(claim/drain) keep LanceDB in sync with the md files under the memory
root. Cascade is the *only* path that writes LanceDB; service / entry
points just write md and trust the daemon to catch up.

Public surface — what lifespan providers / CLI commands import:

- :class:`CascadeOrchestrator` — composite owner; start / stop / sync.
- :class:`CascadeConfig` — construction-time tuning knobs.
- :class:`CascadeAdmin` — system DB schema, index lifecycle and ledger
  operations the server startup and the ``everos cascade`` commands share.
- :data:`KIND_REGISTRY` / :func:`match_kind` — kind dispatch (also
  used by CLI ``cascade sync --path`` to resolve a single file's kind).
- :data:`DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS` — the worker's default
  version-prune retention, shared with the composition root's backfill wiring.

The ``_projections/`` and ``_worker/`` packages are internal; reach them only
through this facade.
"""

from ._worker import (
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS as DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,  # noqa: E501 -- an explicit re-export alias cannot wrap
)
from .admin import CascadeAdmin as CascadeAdmin
from .orchestrator import CascadeConfig as CascadeConfig
from .orchestrator import CascadeHealth as CascadeHealth
from .orchestrator import CascadeOrchestrator as CascadeOrchestrator
from .registry import KIND_REGISTRY as KIND_REGISTRY
from .registry import KindSpec as KindSpec
from .registry import match_kind as match_kind

__all__ = [
    "DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS",
    "KIND_REGISTRY",
    "CascadeAdmin",
    "CascadeConfig",
    "CascadeHealth",
    "CascadeOrchestrator",
    "KindSpec",
    "match_kind",
]
