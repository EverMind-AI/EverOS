"""Cascade worker — drains the ``md_change_state`` queue into the index.

:class:`CascadeWorker` (``_worker``) runs the claim → dispatch → mark cycle
and owns two collaborators: :class:`IndexMaintainer` (``_maintenance``) for
per-kind LanceDB optimize / prune / rebuild, and :class:`LoopSupervisor`
(``_supervision``) for the long-lived loops. Constants and their measured
rationale live in ``_tuning``; the health snapshot in ``_health``.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_HANDLER_CONCURRENCY",
    "DEFAULT_MAX_RETRY",
    "DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS",
    "DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS",
    "DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS",
    "DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS",
    "DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_RETRY_BACKOFF_SECONDS",
    "CascadeWorker",
    "CascadeWorkerHealth",
]

from ._health import CascadeWorkerHealth as CascadeWorkerHealth
from ._tuning import DEFAULT_BATCH_SIZE as DEFAULT_BATCH_SIZE
from ._tuning import DEFAULT_HANDLER_CONCURRENCY as DEFAULT_HANDLER_CONCURRENCY
from ._tuning import DEFAULT_MAX_RETRY as DEFAULT_MAX_RETRY
from ._tuning import (
    DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS as DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
)
from ._tuning import (
    DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS as DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
)
from ._tuning import (
    DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS as DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS,
)
from ._tuning import (
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS as DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,  # noqa: E501 -- an explicit re-export alias cannot wrap
)
from ._tuning import (
    DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS as DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS,  # noqa: E501 -- an explicit re-export alias cannot wrap
)
from ._tuning import DEFAULT_POLL_INTERVAL_SECONDS as DEFAULT_POLL_INTERVAL_SECONDS
from ._tuning import DEFAULT_RETRY_BACKOFF_SECONDS as DEFAULT_RETRY_BACKOFF_SECONDS
from ._worker import CascadeWorker as CascadeWorker
