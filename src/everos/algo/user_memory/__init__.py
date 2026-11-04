"""User-side memory extractors — 4 Extractors + EpisodeReflector + EpisodeSummarizer + boundary re-exports.

:class:`BoundaryDetector` and :class:`DetectionResult` are ``everos.algo.boundary``'s own, re-exported unchanged;
``everos.algo.agent_memory`` re-exports the same class.
"""

from __future__ import annotations

import logging

from everos.algo.boundary import DetectionResult
from everos.algo.user_memory._language import OutputLanguage
from everos.algo.user_memory.atomic_fact import AtomicFactExtractor
from everos.algo.user_memory.boundary import BoundaryDetector
from everos.algo.user_memory.episode import EpisodeExtractor, EpisodeSummarizer
from everos.algo.user_memory.foresight import ForesightExtractor
from everos.algo.user_memory.profile import ProfileExtractor
from everos.algo.user_memory.reflect import EpisodeReflector

__all__ = [
    "AtomicFactExtractor",
    "BoundaryDetector",
    "DetectionResult",
    "EpisodeExtractor",
    "EpisodeReflector",
    "EpisodeSummarizer",
    "ForesightExtractor",
    "OutputLanguage",
    "ProfileExtractor",
]

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
