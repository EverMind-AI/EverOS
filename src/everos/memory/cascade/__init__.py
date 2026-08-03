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
- :class:`BackfillPhase` — dataclass shared between the memory-layer
  phase runners and the CLI's ``PHASES`` copy. The concrete ``PHASES``
  tuple and ``run_backfill`` orchestrator live in
  ``everos.entrypoints.cli.commands._backfill_cmd`` (M11: the memory
  layer must not depend on typer/click).
"""

from ._backfill import BackfillPhase as BackfillPhase
from ._backfill import BackfillPresenter as BackfillPresenter
from ._backfill import NullBackfillPresenter as NullBackfillPresenter
from ._backfill import ome_lock_is_free as ome_lock_is_free
from .orchestrator import CascadeConfig as CascadeConfig
from .orchestrator import CascadeHealth as CascadeHealth
from .orchestrator import CascadeOrchestrator as CascadeOrchestrator
from .registry import KIND_REGISTRY as KIND_REGISTRY
from .registry import KindSpec as KindSpec
from .registry import match_kind as match_kind

__all__ = [
    "KIND_REGISTRY",
    "BackfillPhase",
    "BackfillPresenter",
    "CascadeConfig",
    "CascadeHealth",
    "CascadeOrchestrator",
    "KindSpec",
    "NullBackfillPresenter",
    "match_kind",
    "ome_lock_is_free",
]
