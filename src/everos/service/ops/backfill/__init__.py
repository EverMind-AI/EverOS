"""Backfill -- upgrade memories written before embedding was configured.

Three phases run in order (``vectors`` -> ``clusters`` -> ``skills``), each
in its own private module; ``_deps`` holds the vocabulary they share
(dependencies, presenter, results, summary) and ``_run`` drives the selected
phases until one stops the run. The ops facade reaches all of it through
the names re-exported here.
"""

from __future__ import annotations

__all__ = [
    "BACKFILL_PHASES",
    "BackfillDeps",
    "BackfillOutcome",
    "BackfillPresenter",
    "BackfillSummary",
    "ClustersPhaseResult",
    "NullBackfillPresenter",
    "SkillsPhaseResult",
    "VectorsPhaseResult",
    "format_tokens",
    "run_backfill",
]

from ._deps import BACKFILL_PHASES as BACKFILL_PHASES
from ._deps import BackfillDeps as BackfillDeps
from ._deps import BackfillOutcome as BackfillOutcome
from ._deps import BackfillPresenter as BackfillPresenter
from ._deps import BackfillSummary as BackfillSummary
from ._deps import ClustersPhaseResult as ClustersPhaseResult
from ._deps import NullBackfillPresenter as NullBackfillPresenter
from ._deps import SkillsPhaseResult as SkillsPhaseResult
from ._deps import VectorsPhaseResult as VectorsPhaseResult
from ._deps import format_tokens as format_tokens
from ._run import run_backfill as run_backfill
